# exploration/20_create_plots.py
#
# ZWECK: Alle drei Dual-Axis Plots mit echten Daten erzeugen und speichern.
#
# ABLAUF:
#   1. dual_axis_dataset.csv laden
#   2. Fatigue-Features + Z-Transformation (fuer Clustering)
#   3. Beste k-Means Konfiguration aus cluster_results.csv lesen
#   4. k-Means mit bestem k erneut ausfuehren → Cluster-Labels pro Proband
#   5. Labels in Datensatz einmergen
#   6. Alle drei Plots speichern
#
# AUFRUF:
#   python exploration/20_create_plots.py
#
# VORAUSSETZUNG:
#   python main.py muss vorher gelaufen sein

# ── Imports ──────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))

import config
from fatigue.fatigue_metrics import build_fatigue_feature_table
from extension.viz_plots import dual_axis_snapshot, dual_axis_arrows, cluster_scatter, metrics_table
# ─────────────────────────────────────────────────────────────────────────────


def load_data() -> pd.DataFrame:
    if not config.DUAL_AXIS_CSV.exists():
        raise FileNotFoundError(
            f"CSV nicht gefunden: {config.DUAL_AXIS_CSV}\n"
            "Bitte zuerst main.py ausfuehren."
        )
    df = pd.read_csv(config.DUAL_AXIS_CSV)
    print(f"  Datensatz: {df['Subject'].nunique()} Probanden, {len(df)} Zeilen")
    return df


def load_best_kmeans_k() -> int:
    """Liest die beste k-Means Konfiguration aus cluster_results.csv."""
    if not config.CLUSTER_RESULTS_CSV.exists():
        raise FileNotFoundError(
            f"Cluster-Ergebnisse nicht gefunden: {config.CLUSTER_RESULTS_CSV}\n"
            "Bitte zuerst main.py ausfuehren."
        )

    df_results = pd.read_csv(config.CLUSTER_RESULTS_CSV)

    best_rows = df_results[
        (df_results["method"] == "kmeans") & (df_results["is_best"] == True)
    ]

    if best_rows.empty:
        raise ValueError(
            "Keine beste k-Means Konfiguration in cluster_results.csv gefunden.\n"
            "Pruefe ob RUN_KMEANS=True in config.py gesetzt ist."
        )

    # Bei mehreren is_best-Eintraegen: den mit hoechstem Silhouette-Score waehlen
    best = best_rows.sort_values("silhouette", ascending=False).iloc[0]
    k = int(best["k"])
    print(f"  Beste k-Means Konfiguration: k={k}, Silhouette={best['silhouette']:.4f}")
    return k


def compute_cluster_labels(df: pd.DataFrame, k: int) -> pd.Series:
    """
    Berechnet Fatigue-Features, z-transformiert sie und fuehrt k-Means durch.

    Returns
    -------
    pd.Series mit Index=Subject, Values=Cluster-Label (0-basiert)
    """
    df_features = build_fatigue_feature_table(df)

    feature_cols = [c for c in df_features.columns if c != "Subject"]
    scaler = StandardScaler()
    X_z    = scaler.fit_transform(df_features[feature_cols])

    model  = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=10)
    labels = model.fit_predict(X_z)

    # Labels als 1-basierte Strings: "Cluster 1", "Cluster 2", ...
    label_series = pd.Series(
        [f"Cluster {l + 1}" for l in labels],
        index=df_features["Subject"].values,
        name="cluster_label",
    )
    label_series.index.name = "Subject"

    counts = label_series.value_counts().sort_index()
    print("  Cluster-Groessen:")
    for name, n in counts.items():
        print(f"    {name}: {n} Probanden")

    return label_series


def merge_labels(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mergt Cluster-Labels in den langen Datensatz ein."""
    df_labels = labels.reset_index()
    df_labels.columns = ["Subject", "cluster_label"]
    return df.merge(df_labels, on="Subject", how="left")


def main() -> None:
    print("=" * 55)
    print("Plots erzeugen: Dual-Axis Framework")
    print("=" * 55)

    config.OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1) Daten laden ───────────────────────────────────────────────────────
    print("\nSchritt 1: Daten laden")
    df = load_data()

    # ── 2) Beste k und Cluster-Labels berechnen ──────────────────────────────
    print("\nSchritt 2: Cluster-Labels aus cluster_results.csv")
    k      = load_best_kmeans_k()
    labels = compute_cluster_labels(df, k)
    df     = merge_labels(df, labels)

    # ── 3) Plots speichern ───────────────────────────────────────────────────
    print(f"\nSchritt 3: Plots speichern -> {config.OUTPUT_PLOTS_DIR}")

    print("  Plot 1: dual_axis_snapshot ...")
    dual_axis_snapshot(df, out_dir=config.OUTPUT_PLOTS_DIR)

    print("  Plot 2: dual_axis_arrows ...")
    dual_axis_arrows(df, cluster_col="cluster_label", out_dir=config.OUTPUT_PLOTS_DIR)

    print("  Plot 3: cluster_scatter ...")
    cluster_scatter(df, cluster_col="cluster_label", out_dir=config.OUTPUT_PLOTS_DIR)

    print("  Plot 4: metrics_table ...")
    df_results = pd.read_csv(config.CLUSTER_RESULTS_CSV)
    metrics_table(df_results, out_dir=config.OUTPUT_PLOTS_DIR)

    print("\n" + "=" * 55)
    print(f"Fertig. Plots in: {config.OUTPUT_PLOTS_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
