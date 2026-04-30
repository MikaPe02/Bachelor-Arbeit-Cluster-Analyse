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
# Generate synthetic test data (30 subjects, 5 groups)
python tests/generate_test_data.py

# Run full analysis pipeline (interactive: features → clustering → method selection → plots)
python main.py

# (Standalone) Regenerate plots without re-running the full pipeline
python exploration/20_create_plots.py

# (Real data) Extract DF and SF_norm from MAT files (opens folder-picker dialog)
python extract_to_csv.py
```

## Architecture

### Main Pipeline: `main.py`

Single entry point for the full analysis. All implementation logic lives in src modules — `main.py` contains only function calls. Eight steps:

1. `step1_load_data()` — load `dual_axis_dataset.csv`, validate required columns (incl. `speed_ms`); aborts if `speed_ms` missing
2. `step2_compute_fatigue_features()` — `build_fatigue_feature_table()`; computes Delta + Slope per subject; aborts with `sys.exit(1)` if `< config.MIN_SUBJECTS` subjects. **Not used as clustering input** — saved for fatigue analysis within clusters.
3. `step3_speed_correction()` — `compute_speed_residuals_km1(df)`; linear regression DF ~ speed_ms and SF_norm ~ speed_ms at km 1.0; returns `df_km1` with `DF_residual`, `SF_residual`; saves models to `SPEED_MODELS_PKL`
4. `z_transform(df_km1[["Subject","DF_residual","SF_residual"]])` — StandardScaler on 2 clustering features → `df_style_z`
5. `run_clustering_comparison(df_style_z, config)` — all methods vs. all k, rank aggregation
5b. `elbow_plot(df_results, df_style_z, config)` — normalized WCSS % vs. k
5c. `show_metrics_summary(df_results)` — best config per method/linkage in terminal
5d. `select_clustering(df_results)` — interactive menu: accept recommendation or choose manually
6. `step5_run_final_clustering()` — runs chosen method/k via `run_final_clustering()`
6b. `create_all_plots(df, labels, df_results, config, selection, df_style_z)` — all plots saved to `Outputs/Plots/`; includes `dendrogram_plot()` if `selection["method"] == "hierarchical"`
7. `step6_save_results()` — saves `fatigue_features.csv`, `cluster_results.csv`, `cluster_labels.csv`
8. `sanity_check(df)` — DF and SF_norm range per subject

### Source Library: `Bachelor-Arbeit-Cluster-Analyse/src/`

Scripts add both the project root and this path to `sys.path`:
```python
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))
```

**`fatigue/` package** — core pipeline:
- `io.py` — MAT file loading. Filename is the only source for subject ID and km marker.
- `features.py` — computes DF, SF, SF_norm. `StepFrequency` in MAT files is INT32_MAX (invalid); always recompute from ContactTimes/FlightTimes. Contains `estimate_leg_length(body_height_m)` using De Leva factor 0.53.
- `fatigue_metrics.py` — builds per-subject features: `Delta` (last − first) and `Slope` (linear regression) for DF and SF_norm. Entry point: `build_fatigue_feature_table(df_dual_axis)`. Output columns: `DF_start`, `DF_end`, `Delta_DF`, `Slope_DF`, `SF_start`, `SF_end`, `Delta_SF`, `Slope_SF`. **Saved to CSV, not used as clustering input.**
- `preprocessing.py` — `z_transform(df_features)` (StandardScaler on all non-Subject columns), `sanity_check(df)` (range check per subject).
- `speed_correction.py` — speed-based residual correction. `compute_speed_residuals_km1(df_all)`: filters km==1.0, fits LinearRegression for DF ~ speed_ms and SF_norm ~ speed_ms, returns `df_km1` with `DF_residual`/`SF_residual` columns and a `models` dict. `save_models(models, path)` / `load_models(path)` via pickle. Aborts if `speed_ms` column missing.
- `clustering_eval.py` — k-Means, hierarchical (ward/complete/average/single), HDBSCAN. Entry points: `run_clustering_comparison(df_style_z, cfg)`, `run_final_clustering(df_style_z, selection, random_state)`, `add_best_flag(df_eval)`. Best-k selection via rank aggregation over Silhouette, Davies-Bouldin, Calinski-Harabasz. **Input is always 2-feature style matrix (DF_residual, SF_residual), z-transformed.**
- `clustering_ui.py` — interactive terminal menus. `show_metrics_summary(df_results)`: table of best configs, global best marked with `>`. `select_clustering(df_results)`: method + k selection with input validation, includes HDBSCAN option; returns dict with `method`, `linkage`, `k` (plus `min_cluster_size`/`min_samples` for HDBSCAN). Global best via cross-method re-ranking on absolute metric values; tiebreaker: highest Silhouette.

**`extension/` package** — visualization:
- `viz_plots.py` — all publication-ready plots. Shared constants: `XLIM=(0.45, 0.78)`, `YLIM=(0.65, 1.08)`, `_PALETTE` (Paul Tol), `DPI=300`. Functions: `dual_axis_snapshot()`, `dual_axis_arrows()`, `cluster_scatter()`, `metrics_table()` (booktabs-style; HDBSCAN shown as one row per min_cluster_size, Linkage column shows `mcs=X`, footnote for auto-k), `elbow_plot(df_results, df_features_z, cfg)`, `dendrogram_plot(df_features_z, linkage_method)` (grayscale, cut line, only called for hierarchical), `create_all_plots(df, labels, df_results, cfg, selection, df_features_z)`. Noise points (HDBSCAN label -1) shown in `#AAAAAA` with "Noise (HDBSCAN)" legend entry.

### Configuration: `config.py`

Key settings:
- `DATA_RAW_FOLDER = None` — folder is selected via dialog in `extract_to_csv.py` at runtime
- `SUBJECTS_CSV` — `data/subjects.csv` with columns `Subject`, `leg_length_m`, `body_height_m`, `speed_ms`, `notes`
- `MIN_SUBJECTS = 3` — minimum subjects required before clustering
- `K_RANGE`, `RUN_KMEANS`, `RUN_HIERARCHICAL`, `RUN_HDBSCAN`, `HIERARCHICAL_LINKAGES`
- `RANDOM_STATE = 42`
- `SPEED_MODELS_PKL` — `Outputs/Data/speed_models_km1.pkl`; saved after speed correction in step 3
- Output paths: `FATIGUE_FEATURES_CSV`, `CLUSTER_RESULTS_CSV`, `CLUSTER_LABELS_CSV`, `OUTPUT_PLOTS_DIR`

### Data Flow

```
tests/generate_test_data.py          (synthetic: 30 subjects, 5 van Oeveren groups, incl. speed_ms)
  OR
extract_to_csv.py                    (real MAT files → uses fatigue.io + fatigue.features;
                                      merges speed_ms from data/subjects.csv)
        |
        v
data/processed/dual_axis_dataset.csv   [Subject, km, DF, SF_norm, speed_ms, ...]
data/subjects.csv                      [Subject, leg_length_m, body_height_m, speed_ms, notes]
        |
        v
main.py
  Step 2: fatigue_metrics  →  Delta/Slope per subject (saved only)
  Step 3: speed_correction →  DF_residual, SF_residual at km 1.0
  Step 4: z_transform      →  2-feature style matrix
  Step 5: clustering       →  on DF_residual + SF_residual (running style)
        |
        v
Outputs/Data/fatigue_features.csv    [1 row per subject, Delta/Slope — for fatigue analysis]
Outputs/Data/cluster_results.csv     [1 row per method/k, Silhouette/DB/CH scores, is_best]
Outputs/Data/cluster_labels.csv      [1 row per subject, running style cluster label]
Outputs/Data/speed_models_km1.pkl    [LinearRegression models DF~speed, SF~speed at km 1.0]
Outputs/Plots/elbow_plot.png
Outputs/Plots/dual_axis_snapshot.png
Outputs/Plots/dual_axis_arrows.png
Outputs/Plots/cluster_scatter.png
Outputs/Plots/metrics_table.png
Outputs/Plots/dendrogram_plot.png         (only if hierarchical clustering selected)
```

### MAT File Structure

Files follow `{Subject}_km{KM_INT}_{KM_DEC}.mat` (e.g., `P61_km01_5.mat` = subject P61, km 1.5). Parameters at `PARAMETERS.R.*`; contact/flight times at `CONTACT.<trial_name>.ContactTimes/.FlightTimes`.

### Leg Length

Required for SF normalization: `SF_norm = SF * sqrt(l0/g)`. Priority: (1) `leg_length_m` from `data/subjects.csv`, (2) `estimate_leg_length(body_height_m)` with De Leva factor 0.53, (3) fallback 1.0 m with warning.

### Synthetic Test Data

`tests/generate_test_data.py` generates 30 subjects (P01–P30) across 5 van Oeveren groups (Stick/Bounce/Push/Hop/Sit), each with 19 km-markers and realistic fatigue trends. Overwrites both `dual_axis_dataset.csv` and `data/subjects.csv`. `leg_length_m` is intentionally left empty so `estimate_leg_length()` is exercised. `speed_ms` is generated as N(3.47, 0.33) clipped to [2.78, 4.17] m/s (≈ 10–15 km/h) and merged into `dual_axis_dataset.csv`.

### Raw Data

`.mat` and `.c3d` files are git-ignored. `data/processed/`, `Outputs/`, and `data/subjects.csv` are local only.
