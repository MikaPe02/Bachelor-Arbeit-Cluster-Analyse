# Bachelor-Arbeit-Cluster-Analyse
# Bachelorarbeit – Fatigue Clustering (Biomechanik)

## Setup
- Python via Conda env `fatigue`
- Data files (`.mat`, `.c3d`) are not tracked (see `.gitignore`)

## Workflow
1. Generate whitelist:
   - `python Skribte/05_make_whitelist.py`
2. Build dataset (fatigue features per subject):
   - `python Skribte/06_build_dataset.py`
3. Clustering + validation:
   - `python Skribte/07_scale_and_cluster.py`
   - `python Skribte/08_cluster_validation.py`
4. Export cluster profiles:
   - `python Skribte/09_cluster_profiles_and_export.py`

## Notes
- `whitelist.txt` is frozen feature selection (a-priori).
- Dataset is based on delta + slope across 30 windows.
