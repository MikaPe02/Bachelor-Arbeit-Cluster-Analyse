# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Arbeitsweise
- Erstelle immer zuerst einen Plan und warte auf meine Bestätigung bevor du etwas änderst
- Arbeite in kleinen Schritten – eine Funktion nach der anderen
- Frage nach wenn etwas unklar ist, statt Annahmen zu treffen


## Project Overview

Bachelor's thesis project analyzing running fatigue via biomechanical clustering. Raw motion capture data (`.mat` files from MATLAB) is processed to extract the **Dual-Axis Framework** parameters (Duty Factor and normalized Step Frequency), which are then used for unsupervised clustering of subjects by fatigue patterns.

## Environment

- Conda environment: `fatigue`
- Python executable: `C:\Users\Mika\.conda\envs\fatigue\python.exe`
- Python >=3.11 (uses `list[str] | None` union syntax natively)
- Key dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`
- Optional: `hdbscan` (install via `pip install hdbscan` or `conda install -c conda-forge hdbscan`)

## Running Scripts

All scripts are run from the project root (`C:\Users\Mika\Uni\BA`):

```bash
# Generate synthetic test data (30 subjects, 5 groups)
python tests/generate_test_data.py

# Run full analysis pipeline (interactive: features → clustering → method selection → plots)
python main.py

# (Standalone) Regenerate plots without re-running the full pipeline
python exploration/20_create_plots.py

# (Real data) Extract DF and SF_norm from MAT files
python extract_to_csv.py
```

## Architecture

### Main Pipeline: `main.py`

Single entry point for the full analysis. All implementation logic lives in src modules — `main.py` contains only function calls. Seven steps:

1. `step1_load_data()` — load `dual_axis_dataset.csv`, validate required columns
2. `step2_compute_fatigue_features()` — `build_fatigue_feature_table()`; aborts with `sys.exit(1)` if `< config.MIN_SUBJECTS` subjects
3. `z_transform()` — StandardScaler via `fatigue.preprocessing`
4. `run_clustering_comparison(df_features_z, config)` — all methods vs. all k, rank aggregation
4b. `elbow_plot(df_results, df_features_z, config)` — normalized WCSS % vs. k
4c. `show_metrics_summary(df_results)` — best config per method/linkage in terminal
4d. `select_clustering(df_results)` — interactive menu: accept recommendation or choose manually
5. `step5_run_final_clustering()` — runs chosen method/k via `run_final_clustering()`
5b. `create_all_plots(df, labels, df_results, config)` — all 4 plots saved to `Outputs/Plots/`
6. `step6_save_results()` — saves `fatigue_features.csv`, `cluster_results.csv`, `cluster_labels.csv`
7. `sanity_check(df)` — DF and SF_norm range per subject

### Source Library: `Bachelor-Arbeit-Cluster-Analyse/src/`

Scripts add both the project root and this path to `sys.path`:
```python
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))
```

**`fatigue/` package** — core pipeline:
- `io.py` — MAT file loading. Filename is the only source for subject ID and km marker.
- `features.py` — computes DF, SF, SF_norm. `StepFrequency` in MAT files is INT32_MAX (invalid); always recompute from ContactTimes/FlightTimes. Contains `estimate_leg_length(body_height_m)` using De Leva factor 0.53.
- `fatigue_metrics.py` — builds per-subject features: `Delta` (last − first) and `Slope` (linear regression) for DF and SF_norm. Entry point: `build_fatigue_feature_table(df_dual_axis)`.
- `preprocessing.py` — `z_transform(df_features)` (StandardScaler), `sanity_check(df)` (range check per subject).
- `clustering_eval.py` — k-Means, hierarchical (ward/complete/average/single), HDBSCAN. Entry points: `run_clustering_comparison(df_features_z, cfg)`, `run_final_clustering(df_features_z, selection, random_state)`, `add_best_flag(df_eval)`. Best-k selection via rank aggregation over Silhouette, Davies-Bouldin, Calinski-Harabasz.
- `clustering_ui.py` — interactive terminal menus. `show_metrics_summary(df_results)`: table of best configs, global best marked with →. `select_clustering(df_results)`: method + k selection with input validation. Global best determined by lowest `rank_mean`; fallback to highest Silhouette.

**`extension/` package** — visualization:
- `viz_plots.py` — all publication-ready plots. Shared constants: `XLIM=(0.45, 0.78)`, `YLIM=(0.65, 1.08)`, `_PALETTE` (Paul Tol), `DPI=300`. Functions: `dual_axis_snapshot()`, `dual_axis_arrows()`, `cluster_scatter()`, `metrics_table()` (booktabs-style, black/white/grey), `elbow_plot(df_results, df_features_z, cfg)`, `create_all_plots(df, labels, df_results, cfg)`.

### Configuration: `config.py`

Key settings:
- `DATA_RAW_FOLDER` — path to local `.mat` files (not in git, adjust per machine)
- `SUBJECTS_CSV` — `data/subjects.csv` with columns `Subject`, `leg_length_m`, `body_height_m`
- `MIN_SUBJECTS = 3` — minimum subjects required before clustering
- `K_RANGE`, `RUN_KMEANS`, `RUN_HIERARCHICAL`, `RUN_HDBSCAN`, `HIERARCHICAL_LINKAGES`
- `RANDOM_STATE = 42`
- Output paths: `FATIGUE_FEATURES_CSV`, `CLUSTER_RESULTS_CSV`, `CLUSTER_LABELS_CSV`, `OUTPUT_PLOTS_DIR`

### Data Flow

```
tests/generate_test_data.py          (synthetic: 30 subjects, 5 van Oeveren groups)
  OR
extract_to_csv.py                    (real MAT files → uses fatigue.io + fatigue.features)
        |
        v
data/processed/dual_axis_dataset.csv   [Subject, km, DF, SF_norm, ...]
        |
        v
main.py
        |
        v
Outputs/Data/fatigue_features.csv    [1 row per subject, Delta/Slope features]
Outputs/Data/cluster_results.csv     [1 row per method/k, Silhouette/DB/CH scores, is_best]
Outputs/Data/cluster_labels.csv      [1 row per subject, final cluster assignment]
Outputs/Plots/elbow_plot.png
Outputs/Plots/dual_axis_snapshot.png
Outputs/Plots/dual_axis_arrows.png
Outputs/Plots/cluster_scatter.png
Outputs/Plots/metrics_table.png
```

### MAT File Structure

Files follow `{Subject}_km{KM_INT}_{KM_DEC}.mat` (e.g., `P61_km01_5.mat` = subject P61, km 1.5). Parameters at `PARAMETERS.R.*`; contact/flight times at `CONTACT.<trial_name>.ContactTimes/.FlightTimes`.

### Leg Length

Required for SF normalization: `SF_norm = SF * sqrt(l0/g)`. Priority: (1) `leg_length_m` from `data/subjects.csv`, (2) `estimate_leg_length(body_height_m)` with De Leva factor 0.53, (3) fallback 1.0 m with warning.

### Synthetic Test Data

`tests/generate_test_data.py` generates 30 subjects (P01–P30) across 5 van Oeveren groups (Stick/Bounce/Push/Hop/Sit), each with 19 km-markers and realistic fatigue trends. Overwrites both `dual_axis_dataset.csv` and `data/subjects.csv`. `leg_length_m` is intentionally left empty in subjects.csv so `estimate_leg_length()` is exercised.

### Raw Data

`.mat` and `.c3d` files are git-ignored. `data/processed/`, `Outputs/`, and `data/subjects.csv` are local only.
