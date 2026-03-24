# Skribte/08_cluster_validation.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "Skribte" / "cluster_runs"
DATA_PATH = ROOT / "Skribte" / "fatigue_dataset.csv"

USE_PCA = True
PCA_VAR = 0.90
EMB_NAME = f"pca{int(PCA_VAR*100)}" if USE_PCA else "scaled"

# Load dataset
X = pd.read_csv(DATA_PATH, index_col=0)
X = X.drop(columns=X.columns[X.isna().any()])

if X.shape[0] < 10:
    raise RuntimeError(f"Not enough subjects for validation (n={X.shape[0]}). Need >= 10.")

# Recreate the same embedding used in 07 (so metrics are comparable)
X_scaled = RobustScaler().fit_transform(X)
if USE_PCA:
    X_emb = PCA(n_components=PCA_VAR, random_state=42).fit_transform(X_scaled)
else:
    X_emb = X_scaled

# Load results table produced by 07
results_path = RUN_DIR / "cluster_results.csv"
res = pd.read_csv(results_path)

# Recompute silhouette in a consistent way and compute cluster size stats
rows = []
for _, r in res.iterrows():
    labels_file = RUN_DIR / r["labels_file"]
    labels = pd.read_csv(labels_file, index_col=0).iloc[:, 0].to_numpy()

    # handle noise for HDBSCAN
    mask = labels != -1
    noise_frac = float((~mask).mean())
    labels_use = labels[mask]
    X_use = X_emb[mask]

    # cluster counts
    if mask.sum() > 0:
        counts = pd.Series(labels_use).value_counts()
        min_size = int(counts.min()) if len(counts) else 0
        max_size = int(counts.max()) if len(counts) else 0
    else:
        min_size, max_size = 0, 0

    # silhouette (needs >=2 clusters)
    n_clusters = len(set(labels_use)) if mask.sum() > 0 else 0
    if n_clusters >= 2 and mask.sum() >= 3:
        sil = float(silhouette_score(X_use, labels_use))
    else:
        sil = np.nan

    rows.append(
        {
            "method": r["method"],
            "param": r["param"],
            "embedding": r["embedding"],
            "labels_file": r["labels_file"],
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
            "silhouette_recalc": sil,
            "min_cluster_size": min_size,
            "max_cluster_size": max_size,
        }
    )

val = pd.DataFrame(rows)

# Simple “shortlist” filter (you can tighten later):
# - at least 2 clusters
# - not too tiny clusters (>=3)
# - if HDBSCAN: not too much noise (<=0.30)
short = val[
    (val["n_clusters"] >= 2)
    & (val["min_cluster_size"] >= 3)
    & ((val["method"] != "hdbscan") | (val["noise_frac"] <= 0.30))
].copy()

out_all = RUN_DIR / "cluster_validation.csv"
out_short = RUN_DIR / "cluster_shortlist.csv"
val.to_csv(out_all, index=False)
short.sort_values("silhouette_recalc", ascending=False).to_csv(out_short, index=False)

print("Saved:", out_all)
print("Saved:", out_short)
print("\nTop 10 shortlist by silhouette:")
print(short.sort_values("silhouette_recalc", ascending=False).head(10))
