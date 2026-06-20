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
- Key dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`, `statsmodels`
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

Single entry point for the full analysis. All implementation logic lives in src modules — `main.py` contains only function calls. Steps:

1. `step1_load_data()` — load `dual_axis_dataset.csv`, merge `speed_ms` from `Input/subjects.csv` if missing; aborts if `speed_ms` missing for any subject
2. `step2_compute_fatigue_features()` — `build_fatigue_feature_table()`; computes Delta + Slope per subject; aborts with `sys.exit(1)` if `< config.MIN_SUBJECTS` subjects. **Not used as clustering input** — saved for fatigue analysis within clusters.
3. `step3_select_clustering_input()` — **interactive**: user chooses between (1) speed-corrected residuals (`DF_residual`, `SF_residual` via linear regression at km 1.0) or (2) raw values (`DF`, `SF_norm` at km 1.0). Returns `df_km1` and `feature_cols`. Models saved to `SPEED_MODELS_PKL` only when option 1 is chosen.
4. `z_transform(df_km1[["Subject"] + feature_cols])` — StandardScaler on 2 clustering features → `df_style_z`
5. `run_clustering_comparison(df_style_z, config)` — all methods vs. all k, rank aggregation
5b. `elbow_plot(df_results, df_style_z, config)` — normalized WCSS % vs. k
5c. `show_metrics_summary(df_results)` — best config per method/linkage in terminal
5d. `select_clustering(df_results, df_style_z, config)` — interactive menu: accept recommendation or choose manually; shows elbow + dendrogram before k-selection; afterwards `selection["speed_corrected"]` is set from `feature_cols`
6. `step5_run_final_clustering()` — runs chosen method/k via `run_final_clustering()`
6b. `create_all_plots(df, labels, df_results, config, selection, df_style_z)` — all plots saved to `Outputs/Plots/`; includes `dendrogram_plot()` if `selection["method"] == "hierarchical"`
7. `step6_save_results()` — saves `fatigue_features.csv`, `cluster_results.csv`, `cluster_labels.csv`
7b. `describe_clusters(df, labels, selection, config)` — deskriptive Statistik + ANOVA + Tukey + Boxplots pro Cluster im Terminal + CSV
8. `sanity_check(df)` — DF and SF_norm range per subject
9. `step_fatigue_clustering()` — **optional, interactive**: clusters subjects on fatigue features (Delta_DF, Slope_DF, Delta_SF, Slope_SF); full parallel pipeline with own plots in `Outputs/Plots/Fatigue/` and own CSVs; includes `describe_fatigue_clusters()` for ANOVA + Tukey + Boxplots

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
- `clustering_eval.py` — k-Means, hierarchical (ward/complete/average/single), HDBSCAN. Entry points: `run_clustering_comparison(df_style_z, cfg)`, `run_final_clustering(df_style_z, selection, random_state)`, `add_best_flag(df_eval)`. Best-k selection via rank aggregation over Silhouette, Davies-Bouldin, Calinski-Harabasz. **Input is always z-transformed feature matrix.**
- `clustering_ui.py` — interactive terminal menus. `show_metrics_summary(df_results)`: table of best configs, global best marked with `>`. `select_clustering(df_results, df_style_z, config)`: method + k selection with input validation, includes HDBSCAN option; returns dict with `method`, `linkage`, `k` (plus `min_cluster_size`/`min_samples` for HDBSCAN).

**`extension/` package** — visualization + deskriptive Statistik:
- `viz_plots.py` — all publication-ready plots. Shared constants: `XLIM=(0.43, 0.92)`, `YLIM=(0.62, 1.15)`, `_PALETTE` (Paul Tol colorblind-safe), `DPI=300`, `FIGSIZE=(16/2.54, 13/2.54)`. `_BASE_RCPARAMS`: font.size=11, axes.labelsize=11. All plots are APA-style: no titles (captions go in Word), German axis labels, 300 DPI, min font 11pt.
  - `dual_axis_snapshot(df, selection, out_dir)` — alle Probanden mit Subject-IDs, kein Methodensuffix im Dateinamen
  - `dual_axis_arrows(df, cluster_col, selection, out_dir)` — Pfeile km1→km10 pro Proband, eingefärbt nach Cluster
  - `cluster_scatter(df, cluster_col, selection, out_dir)` — immer Rohdaten DF/SF_norm, unabhängig von Speed-Korrektur
  - `cluster_scatter_residuals(df_features_z, labels, selection, out_dir)` — nur wenn speed_corrected=True
  - `metrics_table(df_results, selection, out_dir)` — APA-Tabelle; gewählte Methode mit `◄ gewählt` markiert; akzeptiert `out_dir` für Unterordner
  - `elbow_plot(df_results, df_features_z, cfg, selection, out_dir)` — normierte WCSS % vs. k; akzeptiert `out_dir`
  - `dendrogram_plot(df_features_z, selection, cfg, labels, out_dir)` — nur bei hierarchischem Clustering; Cluster-Äste in Palettenfarben, oberhalb Schnittlinie grau; Probanden-IDs 45° schräg; Figurbreite 24 cm; akzeptiert `out_dir`
  - `fatigue_cluster_scatter(df_f_z, labels, selection, out_dir)` — Scatter im z-transformierten Fatigue-Feature-Raum
  - `create_all_plots(df, labels, df_results, cfg, selection, df_features_z)` — ruft alle Stil-Clustering-Plots auf
- `descriptive.py` — Deskriptive Statistik + Inferenzstatistik:
  - `describe_clusters(df, labels, selection, cfg)` — für Stil-Clustering: MW/SD/Min/Max pro Cluster für DF, SF_norm, Speed, Anthropometrie + Fatigue-Features; ANOVA + Tukey HSD; Boxplots für Delta/Slope in `Outputs/Plots/Boxplots/`
  - `describe_fatigue_clusters(df_features_raw, labels, selection, cfg)` — für Fatigue-Clustering: gleiche Analyse auf Delta_DF/Slope_DF/Delta_SF/Slope_SF
  - `_test_fatigue_between_clusters()` — ANOVA + Tukey für Feature-Gruppen; speichert `anova_results_{suffix}.csv` und `tukey_results_{suffix}.csv`
  - `_plot_fatigue_boxplots()` — einzelne Boxplots pro Feature in `Outputs/Plots/Boxplots/`; figsize=(16/2.54, 12/2.54); voller Rahmen; kein Signifikanz-Bracket (wird im Text beschrieben)
  - `compare_style_and_fatigue_clusters()` — Kreuztabelle Laufstil- vs. Fatigue-Cluster

### Exploration Scripts: `exploration/`

- `01_speed_correlation_analysis.py` — pre-analysis: correlation between speed and DF/SF_norm
- `02_cluster_speed_independence.py` — post-hoc ANOVA: tests whether clusters differ in speed; saves `Outputs/Plots/cluster_speed_independence.png`
- `03_descriptive_stats.py` — standalone deskriptive Statistik der Stichprobe; speichert `Outputs/Data/descriptive_stats_sample.csv`
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
- `SPEED_MODELS_PKL` — `Outputs/Data/speed_models_km1.pkl`
- Output paths: `FATIGUE_FEATURES_CSV`, `CLUSTER_RESULTS_CSV`, `CLUSTER_LABELS_CSV`, `FATIGUE_CLUSTER_RESULTS_CSV`, `FATIGUE_CLUSTER_LABELS_CSV`, `OUTPUT_PLOTS_DIR`

### Plot Design Conventions

All plots follow APA publication style for Word/Google Docs:
- **Keine Titel** — Beschriftung erfolgt als Bildunterschrift in Word
- **Figurbreite**: 16 cm (`figsize=(16/2.54, h/2.54)`), Dendrogramm 24 cm
- **Schriftgröße**: min. 11pt (axes labels), 10pt (tick labels, legend)
- **DPI**: 300
- **Palette**: Paul Tol colorblind-safe `["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]`
- **Rahmen**: voller Rahmen (alle 4 Spines), Grid mit linewidth=0.5, alpha=0.5
- **Deutsche Achsenbeschriftungen**: "Duty Factor", "Normierte Schrittfrequenz", "Proband", etc.
- **Van-Oeveren-Regionen entfernt** — alle Probanden laufen schnell (Stick/Sit-Bereich), Regionen wären irreführend

### Data Flow

```
extract_to_csv.py                    (real MAT files → uses fatigue.io + fatigue.features;
                                      merges speed_ms from Input/subjects.csv;
                                      writes SF_hz_km1 back into Input/subjects.csv)
        |
        v
Input/processed/dual_axis_dataset.csv   [Subject, km, DF, SF_norm, speed_ms, ...]
Input/subjects.csv                      [Subject, leg_length_m, body_height_m, speed_ms, ...]
        |
        v
main.py  —  Stil-Clustering (DF/SF_norm bei km 1.0)
        |
        v
Outputs/Data/descriptive_stats_sample.csv
Outputs/Data/descriptive_stats_clusters_{method}_{speedflag}.csv
Outputs/Data/fatigue_features.csv          [Delta/Slope pro Proband]
Outputs/Data/cluster_results.csv           [Silhouette/DB/CH je Methode/k]
Outputs/Data/cluster_labels.csv            [Cluster-Label pro Proband]
Outputs/Data/anova_results_{suffix}.csv
Outputs/Data/tukey_results_{suffix}.csv    (nur wenn ANOVA signifikant)
Outputs/Data/speed_models_km1.pkl          (nur bei Residual-Bereinigung)
Outputs/Plots/elbow_plot_{suffix}.png
Outputs/Plots/dual_axis_snapshot.png       (kein Methodensuffix)
Outputs/Plots/dual_axis_arrows_{suffix}.png
Outputs/Plots/cluster_scatter_{suffix}.png
Outputs/Plots/cluster_scatter_residuen_{suffix}.png   (nur speed_corrected)
Outputs/Plots/metrics_table_{speedflag}.png
Outputs/Plots/dendrogram_{suffix}.png      (nur hierarchisch)
Outputs/Plots/Boxplots/boxplot_{feature}_{suffix}.png
Outputs/Plots/cluster_speed_independence.png  (nur via 02_...)

main.py  —  Fatigue-Clustering (Delta/Slope, optional)
        |
        v
Outputs/Data/fatigue_cluster_results.csv
Outputs/Data/fatigue_cluster_labels.csv
Outputs/Data/descriptive_stats_fatigue_clusters_{suffix}.csv
Outputs/Data/anova_results_{suffix}.csv   (Fatigue-Cluster)
Outputs/Plots/Fatigue/elbow_plot_{suffix}.png
Outputs/Plots/Fatigue/metrics_table_{suffix}.png
Outputs/Plots/Fatigue/fatigue_cluster_scatter_{suffix}.png
Outputs/Plots/Fatigue/dendrogram_{suffix}.png  (nur hierarchisch)
Outputs/Plots/Boxplots/boxplot_{feature}_{suffix}.png
```

### MAT File Structure

Files follow `{Subject}_km{KM_INT}_{KM_DEC}.mat` (e.g., `P61_km01_5.mat` = subject P61, km 1.5). Parameters at `PARAMETERS.R.*`; contact/flight times at `CONTACT.<trial_name>.ContactTimes/.FlightTimes`.

### Leg Length

Required for SF normalization: `SF_norm = SF * sqrt(l0/g)`. Priority: (1) `leg_length_m` from `Input/subjects.csv`, (2) `estimate_leg_length(body_height_m)` with De Leva factor 0.53, (3) fallback 1.0 m with warning.

### Raw Data

`.mat` and `.c3d` files are git-ignored. `Input/processed/`, `Outputs/`, and `Input/subjects.csv` are local only.
