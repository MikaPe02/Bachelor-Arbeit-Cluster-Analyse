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
from fatigue.speed_correction import compute_speed_residuals_km1, save_models
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
    """Schritt 2: Fatigue-Features pro Proband berechnen (Delta + Slope).

    Diese Features werden gespeichert und fuer die spaetere Ermüdungsanalyse
    pro Cluster verwendet – aber NICHT als Clustering-Input.
    """
    print("\n=== Schritt 2: Fatigue-Features berechnen ===")

    df_features = build_fatigue_feature_table(df)

    print(f"  Feature-Tabelle: {df_features.shape[0]} Probanden x {df_features.shape[1]} Spalten")
    print(f"  Spalten: {list(df_features.columns)}")
    print(df_features.to_string(index=False))

    return df_features


def step3_speed_correction(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Schritt 3: Speed-bereinigte Residuen bei km 1.0 berechnen.

    Clustering-Input: DF_residual und SF_residual (speed-bereinigt, km 1.0).
    Abbruch wenn speed_ms nicht in der CSV vorhanden ist.
    """
    print("\n=== Schritt 3: Speed-Bereinigung (km 1.0) ===")

    if "speed_ms" not in df.columns:
        raise SystemExit(
            "\nFEHLER: Spalte 'speed_ms' fehlt in dual_axis_dataset.csv.\n"
            "  Loesung: speed_ms in data/subjects.csv eintragen und\n"
            "           extract_to_csv.py erneut ausfuehren."
        )

    df_km1, models = compute_speed_residuals_km1(df)

    config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_models(models, config.SPEED_MODELS_PKL)

    print(f"  Clustering-Features: DF_residual, SF_residual")
    print(f"  Probanden bei km 1.0: {len(df_km1)}")

    return df_km1, models


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

    # Schritt 2: Fatigue-Features (Delta + Slope) – nur fuer Ermüdungsanalyse, nicht Clustering
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

    # Schritt 3: Speed-Bereinigung – Clustering-Input: DF_residual + SF_residual bei km 1.0
    df_km1, _ = step3_speed_correction(df)

    # Schritt 4: Z-Transformation der Clustering-Features
    df_style_z = z_transform(df_km1[["Subject", "DF_residual", "SF_residual"]])

    # Schritt 5: Clustering-Vergleich
    df_results = run_clustering_comparison(df_style_z, config)

    # Schritt 5b: Elbow-Plot
    elbow_plot(df_results, df_style_z, config)

    # Schritt 5c: Metriken-Zusammenfassung
    show_metrics_summary(df_results)

    # Schritt 5d: Konfiguration waehlen
    selection = select_clustering(df_results)

    # Schritt 6: Finales Clustering
    labels = step5_run_final_clustering(df_style_z, selection)

    # Schritt 6b: Plots erstellen
    create_all_plots(df, labels, df_results, config, selection, df_style_z)

    # Schritt 7: Speichern
    step6_save_results(df_features_raw, df_results, labels)

    # Schritt 8: Sanity-Check
    sanity_check(df)

    print("\n" + "=" * 60)
    print("Pipeline abgeschlossen.")
    print("=" * 60)


if __name__ == "__main__":
    main()