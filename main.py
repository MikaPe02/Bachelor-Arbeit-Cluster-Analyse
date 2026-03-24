# main.py
#
# ZWECK: Komplette Analyse-Pipeline in einem Skript.
#        Laedt Dual-Axis Dataset, berechnet Fatigue-Features,
#        fuehrt Clustering durch und speichert alle Ergebnisse.
#
# AUFRUF:
#   conda activate fatigue
#   python main.py
#
# VORAUSSETZUNG:
#   data/processed/dual_axis_dataset.csv muss existieren (erzeugt von extract_to_csv.py)

# ── Imports ──────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Projektpfad so setzen, dass die src-Bibliothek gefunden wird
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))

import config
from fatigue.fatigue_metrics import build_fatigue_feature_table
from fatigue.clustering_eval import evaluate_clustering_methods, add_best_flag
# ─────────────────────────────────────────────────────────────────────────────


def step1_load_data() -> pd.DataFrame:
    """Schritt 1: dual_axis_dataset.csv laden und Pflichtfelder pruefen."""
    print("\n=== Schritt 1: Daten laden ===")

    if not config.DUAL_AXIS_CSV.exists():
        raise FileNotFoundError(
            f"CSV nicht gefunden: {config.DUAL_AXIS_CSV}\n"
            "Bitte zuerst extract_to_csv.py ausfuehren."
        )

    df = pd.read_csv(config.DUAL_AXIS_CSV)

    required_cols = {"Subject", "km", "DF", "SF_norm"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in CSV: {missing}")

    n_subjects = df["Subject"].nunique()
    n_rows = len(df)
    km_values = sorted(df["km"].unique())
    print(f"  Geladen:    {n_rows} Zeilen, {n_subjects} Probanden")
    print(f"  km-Marken:  {km_values}")
    print(f"  Spalten:    {list(df.columns)}")

    return df


def step2_compute_fatigue_features(df: pd.DataFrame) -> pd.DataFrame:
    """Schritt 2: Fatigue-Features pro Proband berechnen (Delta + Slope)."""
    print("\n=== Schritt 2: Fatigue-Features berechnen ===")

    df_features = build_fatigue_feature_table(df)

    print(f"  Feature-Tabelle: {df_features.shape[0]} Probanden x {df_features.shape[1]} Spalten")
    print(f"  Spalten: {list(df_features.columns)}")
    print(df_features.to_string(index=False))

    return df_features


def step3_z_transform(df_features: pd.DataFrame) -> pd.DataFrame:
    """Schritt 3: Z-Transformation aller numerischen Feature-Spalten."""
    print("\n=== Schritt 3: Z-Transformation ===")

    feature_cols = [c for c in df_features.columns if c != "Subject"]

    scaler = StandardScaler()
    X_z = scaler.fit_transform(df_features[feature_cols])

    df_z = pd.DataFrame(X_z, columns=feature_cols)
    df_z.insert(0, "Subject", df_features["Subject"].values)

    means = df_features[feature_cols].mean()
    stds  = df_features[feature_cols].std()
    print("  Mittelwerte (vor Z-Trafo):")
    for col in feature_cols:
        print(f"    {col:15s}: mean={means[col]:+.4f}, std={stds[col]:.4f}")

    return df_z


def step4_clustering(df_features_z: pd.DataFrame) -> pd.DataFrame:
    """Schritt 4: Clustering-Methoden vergleichen (Einstellungen aus config.py)."""
    print("\n=== Schritt 4: Clustering ===")
    print(f"  k_range:     {list(config.K_RANGE)}")
    print(f"  k-Means:     {config.RUN_KMEANS}")
    print(f"  Hierarchisch:{config.RUN_HIERARCHICAL}  Linkages: {config.HIERARCHICAL_LINKAGES}")
    print(f"  HDBSCAN:     {config.RUN_HDBSCAN}")

    df_results = evaluate_clustering_methods(
        df_features_z,
        k_range              = config.K_RANGE,
        include_kmeans       = config.RUN_KMEANS,
        include_hierarchical = config.RUN_HIERARCHICAL,
        include_hdbscan      = config.RUN_HDBSCAN,
        linkages             = config.HIERARCHICAL_LINKAGES,
        hdbscan_min_cluster_sizes = config.HDBSCAN_MIN_CLUSTER_SIZES,
        hdbscan_min_samples       = config.HDBSCAN_MIN_SAMPLES,
        random_state         = config.RANDOM_STATE,
    )

    df_results = add_best_flag(df_results)

    # Beste Konfiguration pro Methode/Linkage ausgeben
    best = df_results[df_results["is_best"]].copy()
    print(f"\n  Ergebnisse: {len(df_results)} Konfigurationen getestet")
    print("  Beste Konfiguration pro Gruppe:")
    display_cols = ["method", "linkage", "k", "silhouette", "davies_bouldin",
                    "calinski_harabasz", "is_best"]
    print(best[display_cols].to_string(index=False))

    return df_results


def step4b_elbow_plot(df_results: pd.DataFrame, df_features_z: pd.DataFrame) -> None:
    """
    Schritt 4b: Elbow-Plot – normierte Innerhalb-Cluster-Streuung gegen k.

    Inertia (Within-Cluster Sum of Squares, WCSS) misst die Gesamtstreuung
    aller Punkte innerhalb ihrer jeweiligen Cluster.
    Bei k=1 liegt alles in einem Cluster → maximale Streuung = 100 %.
    Mit jedem zusaetzlichen Cluster sinkt die Streuung.
    Der 'Knick' (Elbow) markiert den Punkt, ab dem mehr Cluster
    kaum noch zusaetzliche Erklaerungskraft liefern.
    """
    print("\n=== Schritt 4b: Elbow-Plot ===")

    df_km = df_results[df_results["method"] == "kmeans"].copy()
    if df_km.empty:
        print("  Uebersprungen: keine k-Means Ergebnisse (RUN_KMEANS=False).")
        return

    # k=1 als Baseline berechnen (nicht Teil von K_RANGE)
    X = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)
    k1_inertia = float(
        KMeans(n_clusters=1, random_state=config.RANDOM_STATE, n_init=10)
        .fit(X).inertia_
    )

    # k=1 Zeile voranstellen und auf % normieren
    k1_row = pd.DataFrame([{"k": 1, "inertia": k1_inertia, "is_best": False}])
    df_km  = pd.concat([k1_row, df_km], ignore_index=True).sort_values("k")
    df_km["inertia_pct"] = df_km["inertia"] / k1_inertia * 100.0

    best_k = int(df_results.loc[
        (df_results["method"] == "kmeans") & (df_results["is_best"] == True), "k"
    ].iloc[0])

    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    ax.plot(df_km["k"], df_km["inertia_pct"], marker="o", linewidth=1.8,
            color="#4477AA", markersize=6)
    ax.axvline(best_k, linestyle="--", color="#EE6677", linewidth=1.5,
               label=f"Bestes k = {best_k}")
    ax.set_xlabel("Anzahl Cluster k")
    ax.set_ylabel("Innerhalb-Cluster-Streuung\n(% der Streuung bei k=1)")
    ax.set_title("Elbow-Methode: Optimale Clusteranzahl (k-Means)")
    ax.set_xticks(df_km["k"].tolist())
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(frameon=True)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()

    config.OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = config.OUTPUT_PLOTS_DIR / "elbow_plot.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Elbow-Plot gespeichert -> {out}")


def step5_save_results(df_features: pd.DataFrame, df_results: pd.DataFrame) -> None:
    """Schritt 5: Ergebnisse in Outputs/Data/ speichern."""
    print("\n=== Schritt 5: Ergebnisse speichern ===")

    config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_features.to_csv(config.FATIGUE_FEATURES_CSV, index=False)
    print(f"  Fatigue-Features -> {config.FATIGUE_FEATURES_CSV}")

    df_results.to_csv(config.CLUSTER_RESULTS_CSV, index=False)
    print(f"  Cluster-Ergebnisse -> {config.CLUSTER_RESULTS_CSV}")


def step6_sanity_check(df: pd.DataFrame) -> None:
    """
    Schritt 6: Sanity-Check – pruefen ob DF und SF_norm sich zwischen km-Marken unterscheiden.

    Berechnet pro Proband den Wertebereich (max - min) von DF und SF_norm.
    Probanden mit Range = 0 veraendern sich nie → verdaechtig, evtl. Datenfehler.
    """
    print("\n=== Schritt 6: Sanity-Check ===")

    for var in ("DF", "SF_norm"):
        ranges = df.groupby("Subject")[var].agg(lambda s: s.max() - s.min())
        flat   = ranges[ranges == 0.0]

        print(f"\n  {var}:")
        print(f"    Median Range: {ranges.median():.6f}")
        print(f"    Min Range:    {ranges.min():.6f}  (Proband: {ranges.idxmin()})")
        print(f"    Max Range:    {ranges.max():.6f}  (Proband: {ranges.idxmax()})")

        if len(flat) > 0:
            print(f"    WARNUNG: {len(flat)} Proband(en) mit {var}-Range = 0: {list(flat.index)}")
        else:
            print(f"    OK: Alle Probanden zeigen Variation in {var}")


def main() -> None:
    print("=" * 60)
    print("Laufmuedigkeits-Analyse: Dual-Axis Clustering Pipeline")
    print("=" * 60)

    # Schritt 1: Daten laden
    df = step1_load_data()

    # Schritt 2: Fatigue-Features
    df_features_raw = step2_compute_fatigue_features(df)

    n_subjects = len(df_features_raw)
    if n_subjects < config.MIN_SUBJECTS:
        print(
            f"\nFEHLER: Zu wenige Probanden fuer die Analyse.\n"
            f"  Gefunden:  {n_subjects} Proband(en)\n"
            f"  Benoetigt: mindestens {config.MIN_SUBJECTS} (config.MIN_SUBJECTS)\n"
            f"  Loesung:   Weitere MAT-Dateien hinzufuegen und extract_to_csv.py erneut ausfuehren."
        )
        sys.exit(1)

    # Schritt 3: Z-Transformation
    df_features_z = step3_z_transform(df_features_raw)

    # Schritt 4: Clustering
    df_results = step4_clustering(df_features_z)

    # Schritt 4b: Elbow-Plot
    step4b_elbow_plot(df_results, df_features_z)

    # Schritt 5: Speichern
    step5_save_results(df_features_raw, df_results)

    # Schritt 6: Sanity-Check
    step6_sanity_check(df)

    print("\n" + "=" * 60)
    print("Pipeline abgeschlossen.")
    print("=" * 60)


if __name__ == "__main__":
    main()
