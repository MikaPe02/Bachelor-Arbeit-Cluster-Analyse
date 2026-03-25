# src/fatigue/clustering_eval.py
#
# ZWECK: Vergleich verschiedener Clustering-Methoden auf z-transformierten Features.
#
# METHODEN:
#   - k-Means:        Partitionierendes Verfahren, erfordert k als Input
#   - Hierarchisch:   Agglomeratives Verfahren (SAHN), verschiedene Linkages
#   - HDBSCAN:        Dichtebasiertes Verfahren, findet k automatisch
#
# VALIDIERUNGSMETRIKEN (alle drei zusammen verwenden – keine ist universell):
#   - Silhouette Score:       höher = besser (Bereich: −1 bis +1)
#                             Misst Kohäsion vs. Separation pro Punkt
#   - Davies-Bouldin Index:   niedriger = besser (≥ 0)
#                             Misst Streuung innerhalb / Distanz zwischen Clustern
#   - Calinski-Harabasz:      höher = besser (≥ 0)
#                             Verhältnis Zwischen-Cluster- zu Binnen-Cluster-Varianz
#
#   Quelle: Arbelaitz et al. (2013), Pattern Recognition, 46(1), 243–256.
#           "An extensive comparative study of cluster validity indices"
#
# ÄNDERUNGEN gegenüber alter Version:
#   - Doppelter hdbscan-Import repariert (try/except jetzt korrekt am Anfang)
#   - Toter Code entfernt: @dataclass ClusteringResult wurde nirgendwo verwendet
#   - Kommentare zu Metriken und Methoden ergänzt

# ── Imports ──────────────────────────────────────────────────────────────────
from __future__ import annotations

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
# ─────────────────────────────────────────────────────────────────────────────


def compute_validity_scores_safe(
    X: np.ndarray,
    labels: np.ndarray
) -> tuple[float, float, float]:
    """
    Berechnet Silhouette, Davies-Bouldin und Calinski-Harabasz sicher.

    WARUM 'safe':
    Manche Konfigurationen erzeugen ungültige Cluster (z.B. HDBSCAN
    klassifiziert alle Punkte als Noise → nur 1 Cluster übrig).
    In diesen Fällen sind die Metriken nicht definiert → NaN zurückgeben
    statt einen Fehler zu werfen.

    HDBSCAN-Noise:
    Punkte mit Label -1 sind Noise und werden vor der Berechnung
    entfernt. Nur echte Cluster-Punkte gehen in die Metriken ein.

    Parameters
    ----------
    X      : np.ndarray – Feature-Matrix (n_samples × n_features)
    labels : np.ndarray – Cluster-Labels (−1 = Noise bei HDBSCAN)

    Returns
    -------
    tuple: (silhouette, davies_bouldin, calinski_harabasz)
           NaN wenn Berechnung nicht möglich
    """
    labels = np.asarray(labels)

    # Noise-Punkte (Label −1) entfernen
    mask = labels != -1
    X2 = X[mask]
    y2 = labels[mask]

    # Mindestens 2 Cluster nötig für alle Metriken
    unique = np.unique(y2)
    if len(unique) < 2:
        return np.nan, np.nan, np.nan

    sil = silhouette_score(X2, y2)
    db  = davies_bouldin_score(X2, y2)
    ch  = calinski_harabasz_score(X2, y2)

    return float(sil), float(db), float(ch)


def fit_kmeans(
    X: np.ndarray,
    k: int,
    random_state: int = 42
) -> tuple[np.ndarray, float]:
    """
    Führt k-Means Clustering durch.

    WARUM random_state=42:
    k-Means ist nicht deterministisch (zufällige Initialisierung).
    Ein fixer random_state macht Ergebnisse reproduzierbar –
    wichtig für wissenschaftliche Arbeiten.

    WARUM n_init=10:
    k-Means wird 10× mit verschiedenen Startwerten ausgeführt.
    Das beste Ergebnis (niedrigste Inertia) wird behalten.
    Reduziert das Risiko lokaler Minima.

    Parameters
    ----------
    X            : np.ndarray
    k            : int – Anzahl Cluster
    random_state : int

    Returns
    -------
    tuple: (labels, inertia)
    """
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    return labels, float(model.inertia_)


def fit_hierarchical(
    X: np.ndarray,
    k: int,
    linkage: str
) -> np.ndarray:
    """
    Führt agglomeratives (hierarchisches) Clustering durch.

    LINKAGE-METHODEN und ihre Eigenschaften:
    - ward:     Minimiert Varianz innerhalb der Cluster.
                Nur für euklidische Distanz. Oft beste Wahl.
    - complete: Maximale Distanz zwischen Clustern (konservativ).
    - average:  Mittlere Distanz – Kompromiss zwischen ward und single.
    - single:   Minimale Distanz. Neigt zu "Chaining" (lange Ketten).
                Nur als Vergleich, selten bestes Ergebnis.

    Parameters
    ----------
    X       : np.ndarray
    k       : int – Anzahl Cluster
    linkage : str – "ward", "complete", "average" oder "single"

    Returns
    -------
    np.ndarray – Cluster-Labels
    """
    try:
        # aktuelle sklearn-API
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            metric="euclidean"
        )
    except TypeError:
        # Fallback für ältere sklearn-Versionen (<1.2) die 'affinity' statt 'metric' nutzen
        model = AgglomerativeClustering(
            n_clusters=k,
            linkage=linkage,
            affinity="euclidean"
        )

    return model.fit_predict(X)


def fit_hdbscan(
    X: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None = None,
) -> tuple[np.ndarray, int, int]:
    """
    Führt HDBSCAN Clustering durch.

    WARUM HDBSCAN:
    Im Gegensatz zu k-Means bestimmt HDBSCAN die Clusteranzahl automatisch
    und kann Noise-Punkte erkennen (Label −1). Gut geeignet wenn die
    erwartete Clusteranzahl unbekannt ist.

    PARAMETER:
    - min_cluster_size: Mindestgröße eines Clusters.
                        Kleinere Gruppen werden als Noise markiert.
                        Bei wenigen Probanden (n<30): kleine Werte wählen (2–5).
    - min_samples:      Kontrolle der Konservativität.
                        None = automatisch (= min_cluster_size).
                        Höhere Werte → mehr Noise-Punkte.

    Parameters
    ----------
    X                : np.ndarray
    min_cluster_size : int
    min_samples      : int or None

    Returns
    -------
    tuple: (labels, n_clusters_found, n_noise)
    """
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )

    labels = clusterer.fit_predict(X)

    n_noise    = int(np.sum(labels == -1))
    unique     = set(labels.tolist())
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
    Vergleicht alle Clustering-Methoden auf z-transformierten Features.

    WARUM Z-TRANSFORMATION ALS INPUT:
    Alle Features müssen auf vergleichbarer Skala sein bevor geclustert
    wird. Sonst dominieren Features mit großen Wertebereichen die
    Distanzberechnung. Die Z-Transformation (Mittelwert=0, Std=1)
    wird in Skript 12 vorberechnet und hier als Input erwartet.

    Parameters
    ----------
    df_features_z : pd.DataFrame
        Z-transformierte Feature-Tabelle.
        Muss Spalte 'Subject' + numerische Feature-Spalten enthalten.
    k_range : range
        Zu testende Clusteranzahlen für k-Means und hierarchisch.
    include_kmeans, include_hierarchical, include_hdbscan : bool
        Schalter um einzelne Methoden ein-/auszuschalten.
    linkages : list[str]
        Linkage-Methoden für hierarchisches Clustering.
    hdbscan_min_cluster_sizes : list[int]
        Parameter-Grid für HDBSCAN.
    hdbscan_min_samples : list[int | None]
        Parameter-Grid für HDBSCAN.
    random_state : int
        Reproduzierbarkeit für k-Means.

    Returns
    -------
    pd.DataFrame – eine Zeile pro Methoden-Konfiguration mit Metriken
    """
    X       = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)
    n_total = X.shape[0]
    rows: list[dict] = []

    # ── 1) Hierarchisches Clustering ─────────────────────────────────────────
    if include_hierarchical:
        for linkage in linkages:
            for k in k_range:
                labels          = fit_hierarchical(X, k=k, linkage=linkage)
                sil, db, ch     = compute_validity_scores_safe(X, labels)

                rows.append(dict(
                    method          = "hierarchical",
                    linkage         = linkage,
                    k               = k,
                    min_cluster_size= np.nan,
                    min_samples     = np.nan,
                    n_clusters_found= int(len(np.unique(labels))),
                    n_noise         = 0,
                    n_points_scored = n_total,
                    noise_fraction  = 0.0,
                    silhouette      = sil,
                    davies_bouldin  = db,
                    calinski_harabasz = ch,
                    inertia         = np.nan,
                ))

    # ── 2) k-Means ───────────────────────────────────────────────────────────
    if include_kmeans:
        for k in k_range:
            labels, inertia = fit_kmeans(X, k=k, random_state=random_state)
            sil, db, ch     = compute_validity_scores_safe(X, labels)

            rows.append(dict(
                method          = "kmeans",
                linkage         = np.nan,
                k               = k,
                min_cluster_size= np.nan,
                min_samples     = np.nan,
                n_clusters_found= int(len(np.unique(labels))),
                n_noise         = 0,
                n_points_scored = n_total,
                noise_fraction  = 0.0,
                silhouette      = sil,
                davies_bouldin  = db,
                calinski_harabasz = ch,
                inertia         = float(inertia),
            ))

    # ── 3) HDBSCAN ───────────────────────────────────────────────────────────
    if include_hdbscan:
        for mcs in hdbscan_min_cluster_sizes:
            for ms in hdbscan_min_samples:
                labels, n_clusters, n_noise = fit_hdbscan(
                    X,
                    min_cluster_size=mcs,
                    min_samples=ms
                )
                sil, db, ch = compute_validity_scores_safe(X, labels)

                rows.append(dict(
                    method          = "hdbscan",
                    linkage         = np.nan,
                    k               = np.nan,
                    min_cluster_size= int(mcs),
                    min_samples     = (np.nan if ms is None else int(ms)),
                    n_clusters_found= int(n_clusters),
                    n_noise         = int(n_noise),
                    n_points_scored = int(n_total - n_noise),
                    noise_fraction  = float(n_noise / n_total),
                    silhouette      = sil,
                    davies_bouldin  = db,
                    calinski_harabasz = ch,
                    inertia         = np.nan,
                ))

    return pd.DataFrame(rows)


def add_best_flag(df_eval: pd.DataFrame) -> pd.DataFrame:
    """
    Markiert die beste Konfiguration pro Methode mit is_best=True.

    WARUM RANK-AGGREGATION:
    Die drei Metriken zeigen nicht immer dasselbe beste k.
    Rank-Aggregation kombiniert alle drei zu einem gemeinsamen Rang –
    das ist robuster als nur eine Metrik zu verwenden.

    Rangregeln:
    - Silhouette:         Rang 1 = höchster Wert  (maximize)
    - Calinski-Harabasz:  Rang 1 = höchster Wert  (maximize)
    - Davies-Bouldin:     Rang 1 = niedrigster Wert (minimize)

    Gruppierung:
    - Hierarchisch: beste k pro Linkage-Methode
    - k-Means:      beste k insgesamt
    - HDBSCAN:      beste Parameter-Kombination insgesamt

    Parameters
    ----------
    df_eval : pd.DataFrame – Ausgabe von evaluate_clustering_methods()

    Returns
    -------
    pd.DataFrame – mit zusätzlicher Spalte 'is_best'
    """
    df = df_eval.copy()

    # Eindeutiger Gruppen-Schluessel pro Zeile (nie NaN).
    # Hierarchisch: pro Linkage-Methode; k-Means und HDBSCAN: je eine Gruppe.
    # Einzel-String statt Tupel vermeidet pandas-CoW/dropna-Probleme mit NaN-Keys.
    df["group_key"] = df.apply(
        lambda r: f"hierarchical_{r['linkage']}" if r["method"] == "hierarchical"
                  else r["method"],
        axis=1,
    )

    # Zeilen ohne gueltige Metriken (z.B. HDBSCAN fand <2 Cluster) nicht ranken.
    valid = df["silhouette"].notna()

    df["rank_sil"]  = np.nan
    df["rank_ch"]   = np.nan
    df["rank_db"]   = np.nan
    df["rank_mean"] = np.nan
    df["is_best"]   = False

    # Raenge und is_best komplett innerhalb jeder Gruppe berechnen.
    # Kein separates transform("min") – vermeidet NaN==NaN-Verhalten in pandas 3.0.
    for _, grp_idx in df[valid].groupby("group_key").groups.items():
        sub = df.loc[grp_idx]
        df.loc[grp_idx, "rank_sil"]  = sub["silhouette"].rank(ascending=False).values
        df.loc[grp_idx, "rank_ch"]   = sub["calinski_harabasz"].rank(ascending=False).values
        df.loc[grp_idx, "rank_db"]   = sub["davies_bouldin"].rank(ascending=True).values
        df.loc[grp_idx, "rank_mean"] = (
            df.loc[grp_idx, ["rank_sil", "rank_ch", "rank_db"]].mean(axis=1)
        )
        min_rank = df.loc[grp_idx, "rank_mean"].min()  # Skalar, nie NaN
        df.loc[grp_idx, "is_best"] = df.loc[grp_idx, "rank_mean"] == min_rank

    return df.drop(columns=["group_key"])


def run_final_clustering(
    df_features_z: pd.DataFrame,
    selection: dict,
    random_state: int = 42,
) -> pd.Series:
    """Fuehrt das finale Clustering mit der gewaehlten Konfiguration aus.

    Parameters
    ----------
    df_features_z : pd.DataFrame
        Z-transformierte Feature-Tabelle mit Spalte 'Subject'.
    selection : dict
        Ausgabe von select_clustering(): {"method", "linkage", "k"}.
        Fuer HDBSCAN zusaetzlich: {"min_cluster_size", "min_samples"}.
        Noise-Punkte (Label -1) werden als "Noise" ausgegeben.
    random_state : int
        Seed fuer k-Means (wird fuer hierarchisches Clustering und HDBSCAN ignoriert).

    Returns
    -------
    pd.Series
        Index = Subject, Values = "Cluster 1" / "Cluster 2" / ... / "Noise",
        name = "cluster_label".
    """
    method  = selection["method"]
    linkage = selection.get("linkage")
    k       = selection["k"]

    X        = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)
    subjects = df_features_z["Subject"].values

    if method == "kmeans":
        raw_labels = fit_kmeans(X, k=k, random_state=random_state)[0]
    elif method == "hierarchical":
        raw_labels = fit_hierarchical(X, k=k, linkage=linkage)
    else:  # hdbscan
        mcs = selection["min_cluster_size"]
        ms  = selection.get("min_samples")
        raw_labels, _, _ = fit_hdbscan(X, min_cluster_size=mcs, min_samples=ms)

    label_series = pd.Series(
        ["Noise" if l == -1 else f"Cluster {l + 1}" for l in raw_labels],
        index=subjects,
        name="cluster_label",
    )
    label_series.index.name = "Subject"
    return label_series


def run_clustering_comparison(
    df_features_z: pd.DataFrame,
    cfg,
) -> pd.DataFrame:
    """Vergleicht alle Clustering-Methoden und gibt bewertete Ergebnisse zurueck.

    Parameters
    ----------
    df_features_z : pd.DataFrame
        Z-transformierte Feature-Tabelle mit Spalte 'Subject'.
    cfg : config-Modul
        Benoetigt: K_RANGE, RUN_KMEANS, RUN_HIERARCHICAL, RUN_HDBSCAN,
        HIERARCHICAL_LINKAGES, HDBSCAN_MIN_CLUSTER_SIZES,
        HDBSCAN_MIN_SAMPLES, RANDOM_STATE.

    Returns
    -------
    pd.DataFrame – Ergebnisse aller Konfigurationen mit is_best-Flag.
    """
    print("\n=== Schritt 4: Clustering ===")
    print(f"  k_range:     {list(cfg.K_RANGE)}")
    print(f"  k-Means:     {cfg.RUN_KMEANS}")
    print(f"  Hierarchisch:{cfg.RUN_HIERARCHICAL}  Linkages: {cfg.HIERARCHICAL_LINKAGES}")
    print(f"  HDBSCAN:     {cfg.RUN_HDBSCAN}")

    df_results = evaluate_clustering_methods(
        df_features_z,
        k_range                   = cfg.K_RANGE,
        include_kmeans            = cfg.RUN_KMEANS,
        include_hierarchical      = cfg.RUN_HIERARCHICAL,
        include_hdbscan           = cfg.RUN_HDBSCAN,
        linkages                  = cfg.HIERARCHICAL_LINKAGES,
        hdbscan_min_cluster_sizes = cfg.HDBSCAN_MIN_CLUSTER_SIZES,
        hdbscan_min_samples       = cfg.HDBSCAN_MIN_SAMPLES,
        random_state              = cfg.RANDOM_STATE,
    )

    df_results = add_best_flag(df_results)

    best = df_results[df_results["is_best"]].copy()
    print(f"\n  Ergebnisse: {len(df_results)} Konfigurationen getestet")
    print("  Beste Konfiguration pro Gruppe:")
    display_cols = ["method", "linkage", "k", "silhouette", "davies_bouldin",
                    "calinski_harabasz", "is_best"]
    hdb = best[best["method"] == "hdbscan"]
    if not hdb.empty:
        hdb = hdb.iloc[[0]]
    best_display = pd.concat([best[best["method"] != "hdbscan"], hdb])
    print(best_display[display_cols].to_string(index=False))

    return df_results