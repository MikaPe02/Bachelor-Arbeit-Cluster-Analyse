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

# Run full analysis pipeline (features → clustering → elbow plot → sanity check)
python main.py

# Generate all Dual-Axis plots (requires main.py to have run first)
python exploration/20_create_plots.py

# (Legacy) Extract DF and SF_norm from real MAT files
python extract_to_csv.py
```

Scripts in `exploration/` prefixed with `xx_` are deprecated/experimental. Scripts `00_`–`20_` are the active analysis sequence.

## Architecture

### Main Pipeline: `main.py`

Single entry point for the full analysis. Six steps:
1. Load `dual_axis_dataset.csv`
2. Compute fatigue features via `build_fatigue_feature_table()` — aborts with `sys.exit(1)` if `< config.MIN_SUBJECTS` subjects
3. Z-transform with `StandardScaler`
4. Clustering via `evaluate_clustering_methods()` + `add_best_flag()`
4b. Elbow plot (normalized WCSS % vs. k, k=1 baseline computed on the fly)
5. Save `fatigue_features.csv` + `cluster_results.csv` to `Outputs/Data/`
6. Sanity check: DF and SF_norm range per subject

### Source Library: `Bachelor-Arbeit-Cluster-Analyse/src/`

Scripts add both the project root and this path to `sys.path`:
```python
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))
```

**`fatigue/` package** — core pipeline:
- `io.py` — MAT file loading. Filename is the only source for subject ID and km marker.
- `features.py` — computes DF, SF, SF_norm. `StepFrequency` in MAT files is INT32_MAX (invalid); always recompute from ContactTimes/FlightTimes. Contains `estimate_leg_length(body_height_m)` using De Leva factor 0.53.
- `fatigue_metrics.py` — builds per-subject features: `Delta` (last − first) and `Slope` (linear regression) for DF and SF_norm. Entry point: `build_fatigue_feature_table(df_dual_axis)`. **Note:** file was previously truncated at line 208 — the `rows.append({"Subject": s, **metrics})` + `return pd.DataFrame(rows)` lines were added manually.
- `clustering_eval.py` — k-Means, hierarchical (ward/complete/average/single), HDBSCAN. Entry points: `evaluate_clustering_methods(df_features_z, ...)`, `add_best_flag(df_eval)`.

**`extension/` package** — visualization:
- `pca_utils.py` — wraps sklearn PCA.
- `viz.py` — generic scatter/arrow plots for PCA and Dual-Axis space.
- `viz_plots.py` — publication-ready Dual-Axis plots. Three functions: `dual_axis_snapshot()`, `dual_axis_arrows()`, `cluster_scatter()`. All share `XLIM=(0.45, 0.78)` and `YLIM=(0.65, 1.08)`.

### Configuration: `config.py`

Key settings:
- `DATA_RAW_FOLDER` — path to local `.mat` files (not in git, adjust per machine)
- `SUBJECTS_CSV` — `data/subjects.csv` with columns `Subject`, `leg_length_m`, `body_height_m`
- `MIN_SUBJECTS = 3` — minimum subjects required before clustering
- `K_RANGE`, `RUN_KMEANS`, `RUN_HIERARCHICAL`, `RUN_HDBSCAN`, `HIERARCHICAL_LINKAGES`
- `RANDOM_STATE = 42`
- Output paths: `FATIGUE_FEATURES_CSV`, `CLUSTER_RESULTS_CSV`, `OUTPUT_PLOTS_DIR`

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
main.py                              (uses fatigue.fatigue_metrics + fatigue.clustering_eval)
        |
        v
Outputs/Data/fatigue_features.csv    [1 row per subject, Delta/Slope features]
Outputs/Data/cluster_results.csv     [1 row per method/k, Silhouette/DB/CH scores, is_best]
Outputs/Plots/elbow_plot.png
        |
        v
exploration/20_create_plots.py       (uses extension.viz_plots)
        |
        v
Outputs/Plots/dual_axis_snapshot.png
Outputs/Plots/dual_axis_arrows.png
Outputs/Plots/cluster_scatter.png
```

### MAT File Structure

Files follow `{Subject}_km{KM_INT}_{KM_DEC}.mat` (e.g., `P61_km01_5.mat` = subject P61, km 1.5). Parameters at `PARAMETERS.R.*`; contact/flight times at `CONTACT.<trial_name>.ContactTimes/.FlightTimes`.

### Leg Length

Required for SF normalization: `SF_norm = SF * sqrt(l0/g)`. Priority: (1) `leg_length_m` from `data/subjects.csv`, (2) `estimate_leg_length(body_height_m)` with De Leva factor 0.53, (3) fallback 1.0 m with warning.

### Synthetic Test Data

`tests/generate_test_data.py` generates 30 subjects (P01–P30) across 5 van Oeveren groups (Stick/Bounce/Push/Hop/Sit), each with 19 km-markers and realistic fatigue trends. Overwrites both `dual_axis_dataset.csv` and `data/subjects.csv`. `leg_length_m` is intentionally left empty in subjects.csv so `estimate_leg_length()` is exercised.

### Raw Data

`.mat` and `.c3d` files are git-ignored. `data/processed/`, `Outputs/`, and `data/subjects.csv` are local only.
