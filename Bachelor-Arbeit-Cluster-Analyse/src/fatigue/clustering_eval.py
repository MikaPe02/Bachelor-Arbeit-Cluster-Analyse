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


@dataclass
class ClusteringResult:
    method: str
    linkage: Optional[str]
    k: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    inertia: Optional[float]


def compute_validity_scores(X: np.ndarray, labels: np.ndarray) -> Tuple[float, float, float]:
    """
    Higher is better: silhouette, calinski_harabasz
    Lower is better: davies_bouldin
    """
    sil = silhouette_score(X, labels)
    db = davies_bouldin_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
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


def evaluate_clustering_methods(
    df_features_z: pd.DataFrame,
    k_range: range = range(2, 7),
    include_kmeans: bool = True,
    include_hierarchical: bool = True,
    linkages: List[str] = ["single", "average", "complete", "ward"],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate multiple clustering methods on z-transformed features.

    df_features_z: DataFrame with column "Subject" + numeric feature columns.
    Returns a DataFrame with method/linkage/k and validity metrics.
    """
    X = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)

    results: List[ClusteringResult] = []

    if include_hierarchical:
        for linkage in linkages:
            for k in k_range:
                labels = fit_hierarchical(X, k=k, linkage=linkage)
                sil, db, ch = compute_validity_scores(X, labels)

                results.append(
                    ClusteringResult(
                        method="hierarchical",
                        linkage=linkage,
                        k=k,
                        silhouette=sil,
                        davies_bouldin=db,
                        calinski_harabasz=ch,
                        inertia=None,
                    )
                )

    if include_kmeans:
        for k in k_range:
            labels, inertia = fit_kmeans(X, k=k, random_state=random_state)
            sil, db, ch = compute_validity_scores(X, labels)

            results.append(
                ClusteringResult(
                    method="kmeans",
                    linkage=None,
                    k=k,
                    silhouette=sil,
                    davies_bouldin=db,
                    calinski_harabasz=ch,
                    inertia=inertia,
                )
            )

    df_out = pd.DataFrame([r.__dict__ for r in results])
    return df_out


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
    Add a boolean column 'is_best' marking the best row per group:
    group = (method, linkage)

    Ranking rule:
    - silhouette: higher better
    - calinski_harabasz: higher better
    - davies_bouldin: lower better
    -> average rank => smallest is best
    """
    df = df_eval.copy()

    group_cols = ["method", "linkage"]

    df["rank_sil"] = df.groupby(group_cols)["silhouette"].rank(ascending=False)
    df["rank_ch"] = df.groupby(group_cols)["calinski_harabasz"].rank(ascending=False)
    df["rank_db"] = df.groupby(group_cols)["davies_bouldin"].rank(ascending=True)

    df["rank_mean"] = df[["rank_sil", "rank_ch", "rank_db"]].mean(axis=1)

    # beste Zeile pro Gruppe markieren
    df["is_best"] = df["rank_mean"] == df.groupby(group_cols)["rank_mean"].transform("min")

    return df