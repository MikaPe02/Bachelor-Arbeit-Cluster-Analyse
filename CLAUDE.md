# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Arbeitsweise
- Erstelle immer zuerst einen Plan und warte auf meine Bestätigung bevor du etwas änderst
- Arbeite in kleinen Schritten – eine Funktion nach der anderen
- Frage nach wenn etwas unklar ist, statt Annahmen zu treffen
- Zeige mir immer was du ändern möchtest und warte auf meine Bestätigung


## Project Overview

Bachelor's thesis project analyzing running fatigue via biomechanical clustering. Raw motion capture data (`.mat` files from MATLAB) is processed to extract the **Dual-Axis Framework** parameters (Duty Factor and normalized Step Frequency). Subjects are clustered by **running style** (DF_residual, SF_norm_residual at km 1.0, speed-corrected) and then analyzed for fatigue patterns within each style cluster.

## Environment

- Conda environment: `fatigue`
- Python executable: `C:\Users\Mika\.conda\envs\fatigue\python.exe`
- Python >=3.11 (uses `list[str] | None` union syntax natively)
- Key dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`
- Optional: `hdbscan` (install via `pip install hdbscan`); if missing, pipeline runs but selecting HDBSCAN raises a clear `SystemExit` with install instructions

## Running Scripts

All scripts are run from the project root (`C:\Users\Mika\Uni\BA`):

```bash
# Run full analysis pipeline (interactive: features → clustering → method selection → plots)
python main.py

# (Real data) Extract DF and SF_norm from MAT files (opens folder-picker dialog)
# Also writes SF_hz_km1 into Input/subjects.csv
python extract_to_csv.py

# Deskriptive Statistik der Stichprobe (Schritt 0 vor main.py)
python exploration/03_descriptive_stats.py

# (Standalone) Regenerate plots without re-running the full pipeline
python exploration/20_create_plots.py

# Post-hoc validation: speed independence of clusters (ANOVA)
python exploration/02_cluster_speed_independence.py

# Speed correlation analysis (pre-analysis)
python exploration/01_speed_correlation_analysis.py
```

## Architecture

### Main Pipeline: `main.py`

Single entry point for the full analysis. All implementation logic lives in src modules — `main.py` contains only function calls. Eight steps:

1. `step1_load_data()` — load `dual_axis_dataset.csv`, merge `speed_ms` from `Input/subjects.csv` if missing; aborts if `speed_ms` missing for any subject
2. `step2_compute_fatigue_features()` — `build_fatigue_feature_table()`; computes Delta + Slope per subject; aborts with `sys.exit(1)` if `< config.MIN_SUBJECTS` subjects. **Not used as clustering input** — saved for fatigue analysis within clusters.
3. `step3_select_clustering_input()` — **interactive**: user chooses between (1) speed-corrected residuals (`DF_residual`, `SF_residual` via linear regression at km 1.0) or (2) raw values (`DF`, `SF_norm` at km 1.0). Returns `df_km1` and `feature_cols`. Models saved to `SPEED_MODELS_PKL` only when option 1 is chosen.
4. `z_transform(df_km1[["Subject"] + feature_cols])` — StandardScaler on 2 clustering features → `df_style_z`
5. `run_clustering_comparison(df_style_z, config)` — all methods vs. all k, rank aggregation
5b. `elbow_plot(df_results, df_style_z, config)` — normalized WCSS % vs. k; title shows chosen method
5c. `show_metrics_summary(df_results)` — best config per method/linkage in terminal
5d. `select_clustering(df_results, df_style_z, config)` — interactive menu: accept recommendation or choose manually; shows elbow + dendrogram before k-selection; afterwards `selection["speed_corrected"]` is set from `feature_cols`
6. `step5_run_final_clustering()` — runs chosen method/k via `run_final_clustering()`
6b. `create_all_plots(df, labels, df_results, config, selection, df_style_z)` — all plots saved to `Outputs/Plots/`; all plots show chosen method in title; includes `dendrogram_plot()` if `selection["method"] == "hierarchical"`
7. `step6_save_results()` — saves `fatigue_features.csv`, `cluster_results.csv`, `cluster_labels.csv`
7b. `describe_clusters(df, labels, selection, config)` — deskriptive Statistik pro Cluster im Terminal + CSV
8. `sanity_check(df)` — DF and SF_norm range per subject

### Source Library: `Bachelor-Arbeit-Cluster-Analyse/src/`

Scripts add both the project root and this path to `sys.path`:
```python
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))
```

**`fatigue/` package** — core pipeline:
- `io.py` — MAT file loading. Filename is the only source for subject ID and km marker.
- `features.py` — computes DF, SF [Hz], SF_norm. `StepFrequency` in MAT files is INT32_MAX (invalid); always recompute from ContactTimes/FlightTimes via `compute_step_frequency()`. Contains `estimate_leg_length(body_height_m)` using De Leva factor 0.53.
- `fatigue_metrics.py` — builds per-subject features: `Delta` (km 10 − km 1) and `Slope` (linear regression over km 1–10) for DF and SF_norm. Data contains only integer km markers (km 1, 2, ..., 10). Entry point: `build_fatigue_feature_table(df_dual_axis)`. Output columns: `DF_start` (km 1), `DF_end` (km 10), `Delta_DF`, `Slope_DF`, `SF_start` (km 1), `SF_end` (km 10), `Delta_SF`, `Slope_SF`. **Saved to CSV, not used as clustering input.**
- `preprocessing.py` — `z_transform(df_features)` (StandardScaler on all non-Subject columns), `sanity_check(df)` (range check per subject).
- `speed_correction.py` — speed-based residual correction. `compute_speed_residuals_km1(df_all)`: filters km==1.0, fits LinearRegression for DF ~ speed_ms and SF_norm ~ speed_ms, returns `df_km1` with `DF_residual`/`SF_residual` columns and a `models` dict. `save_models(models, path)` / `load_models(path)` via pickle. Aborts if `speed_ms` column missing.
- `clustering_eval.py` — k-Means, hierarchical (ward/complete/average/single), HDBSCAN. Entry points: `run_clustering_comparison(df_style_z, cfg)`, `run_final_clustering(df_style_z, selection, random_state)`, `add_best_flag(df_eval)`. Best-k selection via rank aggregation over Silhouette, Davies-Bouldin, Calinski-Harabasz. **Input is always 2-feature style matrix, z-transformed.**
- `clustering_ui.py` — interactive terminal menus. `show_metrics_summary(df_results)`: table of best configs, global best marked with `>`. `select_clustering(df_results, df_style_z, config)`: method + k selection with input validation, includes HDBSCAN option; shows elbow plot and dendrogram before k-selection; returns dict with `method`, `linkage`, `k` (plus `min_cluster_size`/`min_samples` for HDBSCAN). Global best via cross-method re-ranking on absolute metric values; tiebreaker: highest Silhouette.

**`extension/` package** — visualization + deskriptive Statistik:
- `viz_plots.py` — all publication-ready plots. Shared constants: `XLIM=(0.45, 0.88)`, `YLIM=(0.65, 1.08)`, `_PALETTE` (Paul Tol), `DPI=300`. Helpers: `_method_label(selection)` → readable method string incl. speed_corrected flag; `_file_suffix(selection)` → filename suffix e.g. `_hierarchisch_ward_k3_speedber`. Van Oeveren region labels positioned per conceptual model: Bounce (small DF, mid SF_norm), Hop (mid DF, high SF_norm), Sit (center), Push (mid DF, low SF_norm), Stick (large DF, mid SF_norm). Functions: `dual_axis_snapshot(df, selection, out_dir)` — fixed filename, no method suffix; `dual_axis_arrows(df, cluster_col, selection, out_dir)`; `cluster_scatter(df, cluster_col, selection, out_dir)` — always shows original DF/SF_norm values regardless of speed correction; `cluster_scatter_residuals(df_features_z, labels, selection, out_dir)` — only generated when speed_corrected=True, shows clustering space; `metrics_table(df_results, selection, out_dir)` — filename contains only speed flag; `elbow_plot(df_results, df_features_z, cfg, selection)`; `dendrogram_plot(df_features_z, selection, cfg)` (only for hierarchical); `create_all_plots(df, labels, df_results, cfg, selection, df_features_z)`. All plot titles and filenames include chosen method and speed_corrected flag. Noise points (HDBSCAN label -1) shown in `#AAAAAA`.
- `descriptive.py` — `describe_clusters(df, labels, selection, cfg)`: deskriptive Statistik pro Cluster (DF, SF_norm, Speed, Körpergröße, Gewicht, Beinlänge, SF_hz_km1 falls vorhanden). Verfahrensname im Terminal-Titel. Speichert `Outputs/Data/descriptive_stats_clusters.csv`.

### Exploration Scripts: `exploration/`

- `01_speed_correlation_analysis.py` — pre-analysis: correlation between speed and DF/SF_norm; reads `Input/processed/dual_axis_dataset.csv` and `Input/subjects.csv`
- `02_cluster_speed_independence.py` — post-hoc ANOVA: tests whether clusters differ in speed; reads `Outputs/Data/cluster_labels.csv` and `Outputs/Data/cluster_results.csv` (to display chosen method in plot title); saves `Outputs/Plots/cluster_speed_independence.png`
- `03_descriptive_stats.py` — standalone deskriptive Statistik der Stichprobe (Block 1: subjects.csv — N, M/W, Körpergröße, Gewicht, Beinlänge, Speed, SF_hz_km1; Block 2: DF + SF_norm bei km 1.0); speichert `Outputs/Data/descriptive_stats_sample.csv`
- `10_inspect_mat_structure.py` — inspect raw MAT file structure
- `12_build_cluster_features.py` — standalone feature building
- `14_compare_clustering_methods.py` — standalone clustering comparison
- `16_pca_plot.py` — PCA visualization
- `17_dual_axis_scatter.py` — standalone dual-axis scatter
- `18_dual_axis_arrows_km1_5_to_9_5.py` — standalone arrow plot
- `19_validate_step_frequency.py` — SF validation against MAT data
- `20_create_plots.py` — regenerate all plots from saved CSVs without re-running pipeline

### Configuration: `config.py`

Key settings:
- `DATA_RAW_FOLDER` — `Input/Data`; path to raw MAT files (local, not in git)
- `DATA_PROCESSED_DIR` — `Input/processed`
- `SUBJECTS_CSV` — `Input/subjects.csv` with columns `Subject`, `sex`, `body_height_m`, `dominant_leg`, `leg_length_li_m`, `leg_length_re_m`, `leg_length_m`, `weight_kg`, `speed_ms`, `SF_hz_km1` (last column added by `extract_to_csv.py`)
- `MIN_SUBJECTS = 3` — minimum subjects required before clustering
- `K_RANGE`, `RUN_KMEANS`, `RUN_HIERARCHICAL`, `RUN_HDBSCAN`, `HIERARCHICAL_LINKAGES`
- `RANDOM_STATE = 42`
- `SPEED_MODELS_PKL` — `Outputs/Data/speed_models_km1.pkl`; saved after speed correction in step 3 (only when residual option chosen)
- Output paths: `FATIGUE_FEATURES_CSV`, `CLUSTER_RESULTS_CSV`, `CLUSTER_LABELS_CSV`, `OUTPUT_PLOTS_DIR`

### Data Flow

```
extract_to_csv.py                    (real MAT files → uses fatigue.io + fatigue.features;
                                      merges speed_ms from Input/subjects.csv;
                                      writes SF_hz_km1 back into Input/subjects.csv)
        |
        v
Input/processed/dual_axis_dataset.csv   [Subject, km, DF, SF_norm, speed_ms, ...]
Input/subjects.csv                      [Subject, leg_length_m, body_height_m, speed_ms, notes]
        |
        v
main.py
  Step 2: fatigue_metrics  →  Delta (km10−km1) / Slope (km1–10) per subject (saved only)
  Step 3: user chooses     →  Residuen (DF_residual, SF_residual) ODER Rohdaten (DF, SF_norm)
  Step 4: z_transform      →  2-feature style matrix
  Step 5: clustering       →  on chosen features at km 1.0
        |
        v
Outputs/Data/descriptive_stats_sample.csv   [Stichprobe: MW/SD/Min/Max je Variable]
Outputs/Data/descriptive_stats_clusters.csv [Pro Cluster: MW/SD/Min/Max je Variable]
Outputs/Data/fatigue_features.csv    [1 row per subject, Delta/Slope — for fatigue analysis]
Outputs/Data/cluster_results.csv     [1 row per method/k, Silhouette/DB/CH scores, is_best]
Outputs/Data/cluster_labels.csv      [1 row per subject, running style cluster label]
Outputs/Data/speed_models_km1.pkl    [LinearRegression models DF~speed, SF~speed at km 1.0]
Outputs/Plots/elbow_plot_{method}_{speedflag}.png
Outputs/Plots/dual_axis_snapshot.png                      (fixed name, no method suffix)
Outputs/Plots/dual_axis_arrows_{method}_{speedflag}.png
Outputs/Plots/cluster_scatter_{method}_{speedflag}.png
Outputs/Plots/metrics_table_{speedflag}.png               (only speed flag, no method)
Outputs/Plots/dendrogram_{method}_{speedflag}.png         (only if hierarchical)
Outputs/Plots/cluster_scatter_residuen_{method}_{speedflag}.png  (only if speed_corrected)
Outputs/Plots/cluster_speed_independence.png  (only via 02_cluster_speed_independence.py)
```

### MAT File Structure

Files follow `{Subject}_km{KM_INT}_{KM_DEC}.mat` (e.g., `P61_km01_5.mat` = subject P61, km 1.5). Parameters at `PARAMETERS.R.*`; contact/flight times at `CONTACT.<trial_name>.ContactTimes/.FlightTimes`.

### Leg Length

Required for SF normalization: `SF_norm = SF * sqrt(l0/g)`. Priority: (1) `leg_length_m` from `Input/subjects.csv`, (2) `estimate_leg_length(body_height_m)` with De Leva factor 0.53, (3) fallback 1.0 m with warning.

### Raw Data

`.mat` and `.c3d` files are git-ignored. `Input/processed/`, `Outputs/`, and `Input/subjects.csv` are local only.
