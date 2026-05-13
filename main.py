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

import numpy as np
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
    """Schritt 1: dual_axis_dataset.csv laden, speed_ms aus subjects.csv mergen."""
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

    # speed_ms aus subjects.csv mergen
    if "speed_ms" not in df.columns:
        if not config.SUBJECTS_CSV.exists():
            raise FileNotFoundError(
                f"subjects.csv nicht gefunden: {config.SUBJECTS_CSV}\n"
                "Bitte zuerst extract_to_csv.py ausfuehren."
            )
        subjects = pd.read_csv(config.SUBJECTS_CSV, usecols=["Subject", "speed_ms"])
        df = df.merge(subjects, on="Subject", how="left")
        n_missing_speed = df["speed_ms"].isna().sum()
        if n_missing_speed > 0:
            missing_ids = df[df["speed_ms"].isna()]["Subject"].unique().tolist()
            raise SystemExit(
                f"\nFEHLER: speed_ms fehlt fuer {len(missing_ids)} Proband(en): {missing_ids}\n"
                "  Loesung: speed_ms in data/subjects.csv eintragen."
            )

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


def step3_select_clustering_input(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Schritt 3: Clustering-Input waehlen – Residuen oder Rohdaten bei km 1.0.

    Fragt interaktiv ob Speed-Bereinigung durchgefuehrt werden soll.
    Gibt df_km1 zurueck sowie die Namen der Feature-Spalten fuer Schritt 4.
    """
    print("\n=== Schritt 3: Clustering-Input waehlen ===")
    print()
    print("  Optionen:")
    print("  [1] Residual-Bereinigung  – Clustering auf DF_residual + SF_residual")
    print("      (empfohlen: Speed erklaert bis zu 56% der DF-Varianz)")
    print("  [2] Rohdaten              – Clustering direkt auf DF + SF_norm bei km 1.0")
    print("      (Vergleichsanalyse: zeigt Speed-Gruppen-Effekt)")
    print()

    while True:
        choice = input("  Auswahl [1/2]: ").strip()
        if choice in ("1", "2"):
            break
        print("  Bitte 1 oder 2 eingeben.")

    df_km1 = df[np.abs(df["km"] - 1.0) <= 1e-6].copy()

    if choice == "1":
        print("\n  -> Residual-Bereinigung (speed_ms-Regression bei km 1.0)")
        df_km1, models = compute_speed_residuals_km1(df)
        config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        save_models(models, config.SPEED_MODELS_PKL)
        feature_cols = ["DF_residual", "SF_residual"]
    else:
        print("\n  -> Rohdaten bei km 1.0 (DF + SF_norm, keine Speed-Bereinigung)")
        feature_cols = ["DF", "SF_norm"]

    print(f"  Clustering-Features: {feature_cols}")
    print(f"  Probanden bei km 1.0: {len(df_km1)}")

    return df_km1, feature_cols


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

    # Schritt 3: Clustering-Input waehlen (Residuen oder Rohdaten)
    df_km1, feature_cols = step3_select_clustering_input(df)

    # Schritt 4: Z-Transformation der Clustering-Features
    df_style_z = z_transform(df_km1[["Subject"] + feature_cols])

    # Schritt 5: Clustering-Vergleich
    df_results = run_clustering_comparison(df_style_z, config)

    # Schritt 5b: Elbow-Plot
    elbow_plot(df_results, df_style_z, config)

    # Schritt 5c: Metriken-Zusammenfassung
    show_metrics_summary(df_results)

    # Schritt 5d: Konfiguration waehlen (zeigt Elbow + ggf. Dendrogramm vor k-Abfrage)
    selection = select_clustering(df_results, df_style_z, config)

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