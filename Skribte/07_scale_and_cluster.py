# Skribte/07_run_clustering.py
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Optional HDBSCAN (extra install: conda install -c conda-forge hdbscan)
try:
    import hdbscan  # type: ignore
    HAS_HDBSCAN = True
except Exception:
    HAS_HDBSCAN = False


# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Skribte" / "fatigue_dataset.csv"
OUT_DIR = ROOT / "Skribte" / "cluster_runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------- Config ----------------
USE_PCA = True
PCA_VAR = 0.90  # fixed a-priori for BA
K_RANGE = range(2, 9)

RUN_KMEANS = True

RUN_HIER = True
# Note: 'single' tends to chaining; keep it for comparison, but interpret carefully
HIER_LINKAGES = ["ward", "average", "complete", "single"]

RUN_HDBSCAN = True
HDBSCAN_PARAMS = [
    {"min_cluster_size": 5, "min_samples": None},
    {"min_cluster_size": 8, "min_samples": None},
    {"min_cluster_size": 10, "min_samples": None},
]
# --------------------------------------


def compute_metrics(X: np.ndarray, labels: np.ndarray) -> dict:
    """
    Computes clustering metrics.
    If labels contain noise (-1), excludes those points for metric computation and reports noise_frac.
    """
    labels = np.asarray(labels)
    mask = labels != -1
    noise_frac = float((~mask).mean())

    if mask.sum() < 3:
        return {
            "n_clusters": np.nan,
            "noise_frac": noise_frac,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    lab = labels[mask]
    X_use = X[mask]

    n_clusters = len(set(lab))
    if n_clusters < 2:
        return {
            "n_clusters": n_clusters,
            "noise_frac": noise_frac,
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return {
        "n_clusters": n_clusters,
        "noise_frac": noise_frac,
        "silhouette": float(silhouette_score(X_use, lab)),
        "calinski_harabasz": float(calinski_harabasz_score(X_use, lab)),
        "davies_bouldin": float(davies_bouldin_score(X_use, lab)),
    }


def save_labels(index: pd.Index, labels: np.ndarray, name: str) -> str:
    out = OUT_DIR / f"labels_{name}.csv"
    pd.Series(labels, index=index, name="cluster").to_csv(out)
    return out.name


# ---------------- Load + preprocess ----------------
X = pd.read_csv(DATA_PATH, index_col=0)

if X.shape[0] < 10:
    raise RuntimeError(f"Not enough subjects for clustering (n={X.shape[0]}). Need >= 10.")

# strict start: drop columns with any NaNs
X = X.drop(columns=X.columns[X.isna().any()])

# scale
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# embedding
if USE_PCA:
    pca = PCA(n_components=PCA_VAR, random_state=42)
    X_emb = pca.fit_transform(X_scaled)
    emb_name = f"pca{int(PCA_VAR*100)}"

    # save PCA variance info for reporting
    var_ratio = pca.explained_variance_ratio_
    pd.Series(var_ratio).to_csv(OUT_DIR / f"pca_variance_ratio_{emb_name}.csv", index=False)
    meta = {
        "n_subjects": X.shape[0],
        "n_features_in": X.shape[1],
        "n_components": X_emb.shape[1],
        "target_variance": PCA_VAR,
        "achieved_variance": float(var_ratio.sum()),
    }
    pd.Series(meta).to_csv(OUT_DIR / f"pca_meta_{emb_name}.csv")
else:
    X_emb = X_scaled
    emb_name = "scaled"

print("Data shape:", X.shape, "| Embedding:", X_emb.shape, "|", emb_name)

# ---------------- Run methods ----------------
rows: list[dict] = []

# KMeans
if RUN_KMEANS:
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init="auto", random_state=42)
        labels = model.fit_predict(X_emb)
        metrics = compute_metrics(X_emb, labels)
        rows.append(
            {
                "method": "kmeans",
                "param": f"k={k}",
                "embedding": emb_name,
                **metrics,
                "labels_file": save_labels(X.index, labels, f"kmeans_k{k}_{emb_name}"),
            }
        )

# Agglomerative
if RUN_HIER:
    for linkage in HIER_LINKAGES:
        for k in K_RANGE:
            # ward is defined for euclidean distances; on PCA space this is fine
            model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
            labels = model.fit_predict(X_emb)
            metrics = compute_metrics(X_emb, labels)
            rows.append(
                {
                    "method": "agglomerative",
                    "param": f"linkage={linkage},k={k}",
                    "embedding": emb_name,
                    **metrics,
                    "labels_file": save_labels(X.index, labels, f"agglo_{linkage}_k{k}_{emb_name}"),
                }
            )

# HDBSCAN
if RUN_HDBSCAN:
    if not HAS_HDBSCAN:
        print("HDBSCAN not installed. Install via: conda install -c conda-forge hdbscan")
    else:
        for p in HDBSCAN_PARAMS:
            model = hdbscan.HDBSCAN(**p)
            labels = model.fit_predict(X_emb)
            metrics = compute_metrics(X_emb, labels)
            rows.append(
                {
                    "method": "hdbscan",
                    "param": ",".join([f"{k}={v}" for k, v in p.items()]),
                    "embedding": emb_name,
                    **metrics,
                    "labels_file": save_labels(X.index, labels, f"hdbscan_mcs{p['min_cluster_size']}_{emb_name}"),
                }
            )

res = pd.DataFrame(rows)

out_csv = OUT_DIR / "cluster_results.csv"
res.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print("\nTop 15 by silhouette (per method, overall):")
print(res.sort_values(["method", "silhouette"], ascending=[True, False]).groupby("method").head(5))
