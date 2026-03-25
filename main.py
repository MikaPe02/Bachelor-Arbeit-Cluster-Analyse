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

import pandas as pd

# Projektpfad so setzen, dass die src-Bibliothek gefunden wird
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))

import config
from fatigue.fatigue_metrics import build_fatigue_feature_table
from fatigue.preprocessing import z_transform, sanity_check
from fatigue.clustering_eval import run_clustering_comparison, run_final_clustering
from fatigue.clustering_ui import show_metrics_summary, select_clustering
from extension.viz_plots import elbow_plot, create_all_plots
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


def step5_run_final_clustering(df_features_z: pd.DataFrame, selection: dict) -> pd.Series:
    """Schritt 5: Finale Clustering-Ausfuehrung mit der gewaehlten Konfiguration."""
    method  = selection["method"]
    linkage = selection.get("linkage")
    k       = selection["k"]

    print(f"\n=== Schritt 5: Finales Clustering ===")
    linkage_label = f", Linkage={linkage}" if linkage else ""
    print(f"  Methode: {method}{linkage_label}, k={k}")

    label_series = run_final_clustering(df_features_z, selection, random_state=config.RANDOM_STATE)

    counts = label_series.value_counts().sort_index()
    print("  Cluster-Groessen:")
    for name, n in counts.items():
        print(f"    {name}: {n} Probanden")

    return label_series


def step6_save_results(
    df_features: pd.DataFrame,
    df_results: pd.DataFrame,
    labels: pd.Series,
) -> None:
    """Schritt 6: Ergebnisse in Outputs/Data/ speichern."""
    print("\n=== Schritt 6: Ergebnisse speichern ===")

    config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df_features.to_csv(config.FATIGUE_FEATURES_CSV, index=False)
    print(f"  Fatigue-Features -> {config.FATIGUE_FEATURES_CSV}")

    df_results.to_csv(config.CLUSTER_RESULTS_CSV, index=False)
    print(f"  Cluster-Ergebnisse -> {config.CLUSTER_RESULTS_CSV}")

    df_labels = labels.reset_index()
    df_labels.columns = ["Subject", "cluster_label"]
    df_labels.to_csv(config.CLUSTER_LABELS_CSV, index=False)
    print(f"  Cluster-Labels -> {config.CLUSTER_LABELS_CSV}")


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
    df_features_z = z_transform(df_features_raw)

    # Schritt 4: Clustering
    df_results = run_clustering_comparison(df_features_z, config)

    # Schritt 4b: Elbow-Plot
    elbow_plot(df_results, df_features_z, config)

    # Schritt 4c: Metriken-Zusammenfassung
    show_metrics_summary(df_results)

    # Schritt 4d: Konfiguration waehlen
    selection = select_clustering(df_results)

    # Schritt 5: Finales Clustering
    labels = step5_run_final_clustering(df_features_z, selection)

    # Schritt 5b: Plots erstellen
    create_all_plots(df, labels, df_results, config, selection, df_features_z)

    # Schritt 6: Speichern
    step6_save_results(df_features_raw, df_results, labels)

    # Schritt 7: Sanity-Check
    sanity_check(df)

    print("\n" + "=" * 60)
    print("Pipeline abgeschlossen.")
    print("=" * 60)


if __name__ == "__main__":
    main()