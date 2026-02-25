from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
import hdbscan
try:
    import hdbscan
except ImportError:
    hdbscan = None


@dataclass
class ClusteringResult:
    method: str
    linkage: Optional[str]
    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    inertia: Optional[float]


def compute_validity_scores_safe(X: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """
    Compute validity scores safely.
    Returns NaN if scores are not defined (e.g., <2 clusters after filtering noise).

    For HDBSCAN: labels can contain -1 (noise). Noise points are excluded for scoring.
    """
    labels = np.asarray(labels)

    # Noise entfernen (-1)
    mask = labels != -1
    X2 = X[mask]
    y2 = labels[mask]

    # mind. 2 Cluster nötig
    unique = np.unique(y2)
    if len(unique) < 2:
        return np.nan, np.nan, np.nan

    sil = silhouette_score(X2, y2)
    db = davies_bouldin_score(X2, y2)
    ch = calinski_harabasz_score(X2, y2)
    return sil, db, ch

def fit_hierarchical(X: np.ndarray, k: int, linkage: str) -> np.ndarray:
    """
    Fit agglomerative clustering (SAHN) for a given linkage and return labels.
    """
    try:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric="euclidean")
    except TypeError:
        model = AgglomerativeClustering(n_clusters=k, linkage=linkage, affinity="euclidean")

    labels = model.fit_predict(X)
    return labels


def fit_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> Tuple[np.ndarray, float]:
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, float(model.inertia_)

def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    Fit HDBSCAN and return:
    - labels
    - n_clusters_found (excluding noise label -1)
    - n_noise
    """
    if hdbscan is None:
        raise ImportError("Package 'hdbscan' is not installed. Install with: pip install hdbscan")

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(X)

    n_noise = int(np.sum(labels == -1))
    unique = set(labels.tolist())
    n_clusters = int(len(unique) - (1 if -1 in unique else 0))

    return labels, n_clusters, n_noise


def evaluate_clustering_methods(
    df_features_z: pd.DataFrame,
    k_range: range = range(2, 7),
    include_kmeans: bool = True,
    include_hierarchical: bool = True,
    include_hdbscan: bool = True,
    linkages: list[str] = ["single", "average", "complete", "ward"],
    hdbscan_min_cluster_sizes: list[int] = [3, 4, 5, 6],
    hdbscan_min_samples: list[int | None] = [None, 2, 3, 4],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate hierarchical (multiple linkages), kmeans (k=2..6), and HDBSCAN (parameter grid)
    on z-transformed features.

    Returns one combined DataFrame with consistent columns.
    """

    X = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)

    rows: list[dict] = []

    # 1) Hierarchical
    if include_hierarchical:
        for linkage in linkages:
            for k in k_range:
                labels = fit_hierarchical(X, k=k, linkage=linkage)

                sil, db, ch = compute_validity_scores_safe(X, labels)

                rows.append(
                    dict(
                        method="hierarchical",
                        linkage=linkage,
                        k=k,
                        min_cluster_size=np.nan,
                        min_samples=np.nan,
                        n_clusters_found=int(len(np.unique(labels))),
                        n_noise=0,
                        silhouette=sil,
                        davies_bouldin=db,
                        calinski_harabasz=ch,
                        inertia=np.nan,
                    )
                )

    # 2) KMeans
    if include_kmeans:
        for k in k_range:
            labels, inertia = fit_kmeans(X, k=k, random_state=random_state)

            sil, db, ch = compute_validity_scores_safe(X, labels)

            rows.append(
                dict(
                    method="kmeans",
                    linkage=np.nan,
                    k=k,
                    min_cluster_size=np.nan,
                    min_samples=np.nan,
                    n_clusters_found=int(len(np.unique(labels))),
                    n_noise=0,
                    silhouette=sil,
                    davies_bouldin=db,
                    calinski_harabasz=ch,
                    inertia=float(inertia),
                )
            )

    # 3) HDBSCAN (kein k, sondern Parameter)
    if include_hdbscan:
        for mcs in hdbscan_min_cluster_sizes:
            for ms in hdbscan_min_samples:
                labels, n_clusters, n_noise = fit_hdbscan(X, min_cluster_size=mcs, min_samples=ms)

                sil, db, ch = compute_validity_scores_safe(X, labels)

                rows.append(
                    dict(
                        method="hdbscan",
                        linkage=np.nan,
                        k=np.nan,
                        min_cluster_size=int(mcs),
                        min_samples=(np.nan if ms is None else int(ms)),
                        n_clusters_found=int(n_clusters),
                        n_noise=int(n_noise),
                        silhouette=sil,
                        davies_bouldin=db,
                        calinski_harabasz=ch,
                        inertia=np.nan,
                    )
                )

    return pd.DataFrame(rows)


def suggest_best_k(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Suggest best k per (method, linkage) using rank aggregation:
    - maximize silhouette
    - maximize calinski_harabasz
    - minimize davies_bouldin
    """
    df = df_eval.copy()

    df["rank_sil"] = df.groupby(["method", "linkage"])["silhouette"].rank(ascending=False)
    df["rank_ch"] = df.groupby(["method", "linkage"])["calinski_harabasz"].rank(ascending=False)
    df["rank_db"] = df.groupby(["method", "linkage"])["davies_bouldin"].rank(ascending=True)

    df["rank_mean"] = df[["rank_sil", "rank_ch", "rank_db"]].mean(axis=1)

    best_rows = (
        df.sort_values(["method", "linkage", "rank_mean"])
          .groupby(["method", "linkage"], as_index=False)
          .first()
    )

    best_rows["k_minus_1"] = best_rows["k"] - 1
    best_rows["k_plus_1"] = best_rows["k"] + 1

    return best_rows[
        ["method", "linkage", "k", "k_minus_1", "k_plus_1",
         "silhouette", "davies_bouldin", "calinski_harabasz", "rank_mean"]
    ]

def add_best_flag(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_best=True for the best setting per:
    - hierarchical: per linkage
    - kmeans: overall
    - hdbscan: overall

    Uses rank aggregation of silhouette (max), CH (max), DB (min).
    Ignores rows with NaN silhouette (e.g. HDBSCAN found <2 clusters).
    """
    df = df_eval.copy()

    # Gruppierung: hierarchical nach linkage, sonst nur nach method
    df["group_linkage"] = df["linkage"]
    df.loc[df["method"] != "hierarchical", "group_linkage"] = "none"

    group_cols = ["method", "group_linkage"]

    # NaN rows (nicht bewertbar) sollen nicht "best" werden
    valid = df["silhouette"].notna()

    df["rank_sil"] = np.nan
    df["rank_ch"] = np.nan
    df["rank_db"] = np.nan
    df["rank_mean"] = np.nan

    df.loc[valid, "rank_sil"] = df[valid].groupby(group_cols)["silhouette"].rank(ascending=False)
    df.loc[valid, "rank_ch"] = df[valid].groupby(group_cols)["calinski_harabasz"].rank(ascending=False)
    df.loc[valid, "rank_db"] = df[valid].groupby(group_cols)["davies_bouldin"].rank(ascending=True)

    df.loc[valid, "rank_mean"] = df.loc[valid, ["rank_sil", "rank_ch", "rank_db"]].mean(axis=1)

    df["is_best"] = False
    df.loc[valid, "is_best"] = df.loc[valid, "rank_mean"] == df.loc[valid].groupby(group_cols)["rank_mean"].transform("min")

    return df.drop(columns=["group_linkage"])