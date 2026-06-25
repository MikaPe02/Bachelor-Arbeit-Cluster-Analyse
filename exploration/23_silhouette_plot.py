# exploration/23_silhouette_plot.py
#
# ZWECK: Silhouette-Wert pro Proband berechnen und als Balkendiagramm speichern.
#        Zeigt welche Probanden gut/schlecht ihrem Cluster zugeordnet sind.
#        Liest cluster_labels.csv und dual_axis_dataset.csv — kein Pipeline-Lauf noetig.
#
# AUFRUF:
#   python exploration/23_silhouette_plot.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import silhouette_samples

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import config

# ── Konstanten ────────────────────────────────────────────────────────────────

_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

_RCPARAMS = {
    "font.size":        11,
    "axes.labelsize":   11,
    "xtick.labelsize":   9,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
}

FIGSIZE = (16 / 2.54, 13 / 2.54)
DPI     = 300


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _find_latest(pattern: str) -> Path | None:
    files = sorted(
        config.OUTPUT_DATA_DIR.glob(pattern),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _load_feature_matrix(df_labels: pd.DataFrame) -> pd.DataFrame | None:
    """
    Laedt die z-transformierten Clustering-Features.
    Versucht speed_models_pkl zu lesen um zu bestimmen ob Residuen oder Rohdaten.
    Gibt DataFrame mit Subject + Feature-Spalten zurueck.
    """
    if not config.DUAL_AXIS_CSV.exists():
        return None

    df_long = pd.read_csv(config.DUAL_AXIS_CSV)

    # speed_ms mergen
    if "speed_ms" not in df_long.columns and config.SUBJECTS_CSV.exists():
        subj = pd.read_csv(config.SUBJECTS_CSV, usecols=["Subject", "speed_ms"])
        df_long = df_long.merge(subj, on="Subject", how="left")

    df_km1 = df_long[np.abs(df_long["km"] - 1.0) <= 1e-6].copy()

    # Residuen-Modelle laden falls vorhanden
    if config.SPEED_MODELS_PKL.exists():
        try:
            import pickle
            with open(config.SPEED_MODELS_PKL, "rb") as f:
                models = pickle.load(f)
            df_km1["DF_residual"]  = df_km1["DF"]  - models["DF"].predict(df_km1[["speed_ms"]])
            df_km1["SF_residual"]  = df_km1["SF_norm"] - models["SF_norm"].predict(df_km1[["speed_ms"]])
            feature_cols = ["DF_residual", "SF_residual"]
            print("  Feature-Matrix: DF_residual + SF_residual (speed-bereinigt)")
        except Exception:
            feature_cols = ["DF", "SF_norm"]
            print("  Feature-Matrix: DF + SF_norm (Rohdaten, Residuen-Laden fehlgeschlagen)")
    else:
        feature_cols = ["DF", "SF_norm"]
        print("  Feature-Matrix: DF + SF_norm (Rohdaten)")

    # Nur Probanden die auch in Labels vorhanden sind
    subjects = df_labels["Subject"].tolist()
    df_km1 = df_km1[df_km1["Subject"].isin(subjects)].copy()

    # Z-Transformation
    from sklearn.preprocessing import StandardScaler
    X = df_km1[feature_cols].values
    X_z = StandardScaler().fit_transform(X)
    df_z = pd.DataFrame(X_z, columns=feature_cols)
    df_z.insert(0, "Subject", df_km1["Subject"].values)

    return df_z


# ── Plot ──────────────────────────────────────────────────────────────────────

def silhouette_plot(
    df_z: pd.DataFrame,
    df_labels: pd.DataFrame,
    suffix: str,
    out_dir: Path,
) -> None:
    """
    Silhouette-Balkendiagramm: ein Balken pro Proband,
    sortiert nach Cluster und Silhouette-Wert, eingefaerbt nach Cluster.
    """
    label_col = "cluster_label"

    # Feature-Matrix und Labels zusammenfuehren
    feature_cols = [c for c in df_z.columns if c != "Subject"]
    df = df_z.merge(df_labels[["Subject", label_col]], on="Subject", how="inner")

    X      = df[feature_cols].values
    labels = df[label_col].values

    # Silhouette-Werte berechnen
    sil_vals = silhouette_samples(X, labels)
    df["silhouette"] = sil_vals

    mean_sil = sil_vals.mean()
    print(f"\n  Mittlerer Silhouette-Wert: {mean_sil:.3f}")

    # Sortierung: erst nach Cluster, dann innerhalb absteigend nach Silhouette
    cluster_ids = sorted(df[label_col].unique(), key=str)
    df_sorted = pd.concat([
        df[df[label_col] == cid].sort_values("silhouette", ascending=False)
        for cid in cluster_ids
    ]).reset_index(drop=True)

    # Terminal-Ausgabe
    print(f"\n  {'Proband':<12} {'Cluster':<12} {'Silhouette':>12}")
    print("  " + "-" * 38)
    for _, row in df_sorted.iterrows():
        flag = "  <-- Grenzfall" if row["silhouette"] < 0.2 else ""
        print(f"  {str(row['Subject']):<12} {str(row[label_col]):<12} {row['silhouette']:>12.3f}{flag}")

    # Grenzfaelle zusammenfassen
    borderline = df_sorted[df_sorted["silhouette"] < 0.2]
    if len(borderline) > 0:
        print(f"\n  Grenzfaelle (s < 0.2): {borderline['Subject'].tolist()}")
    else:
        print("\n  Keine Grenzfaelle (alle s >= 0.2)")

    # CSV: alle Probanden
    csv_path = config.OUTPUT_DATA_DIR / f"silhouette_values{suffix}.csv"
    df_sorted[["Subject", label_col, "silhouette"]].to_csv(csv_path, index=False)
    print(f"\n  -> Silhouette-Werte gespeichert: {csv_path}")

    # APA-Tabelle: MW, SD, Min, Max pro Cluster + Gesamtmittelwert
    summary_rows = []
    for cid in cluster_ids:
        vals = df_sorted.loc[df_sorted[label_col] == cid, "silhouette"]
        summary_rows.append({
            "Cluster":    str(cid),
            "N":          len(vals),
            "MW":         round(vals.mean(), 3),
            "SD":         round(vals.std(ddof=1), 3),
            "Min":        round(vals.min(), 3),
            "Max":        round(vals.max(), 3),
            "Grenzfälle (s<0.2)": int((vals < 0.2).sum()),
        })
    summary_rows.append({
        "Cluster":    "Gesamt",
        "N":          len(df_sorted),
        "MW":         round(mean_sil, 3),
        "SD":         round(df_sorted["silhouette"].std(ddof=1), 3),
        "Min":        round(df_sorted["silhouette"].min(), 3),
        "Max":        round(df_sorted["silhouette"].max(), 3),
        "Grenzfälle (s<0.2)": int((df_sorted["silhouette"] < 0.2).sum()),
    })
    df_summary = pd.DataFrame(summary_rows)
    summary_path = config.OUTPUT_DATA_DIR / f"silhouette_summary{suffix}.csv"
    df_summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    print(f"  -> Silhouette-Zusammenfassung: {summary_path}")
    print(df_summary.to_string(index=False))

    # Vereinfachter Plot: MW ± SD pro Cluster als Balken
    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    x_pos = range(len(cluster_ids))
    mws = [df_sorted.loc[df_sorted[label_col] == cid, "silhouette"].mean() for cid in cluster_ids]
    sds = [df_sorted.loc[df_sorted[label_col] == cid, "silhouette"].std(ddof=1) for cid in cluster_ids]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cluster_ids))]

    ax.bar(x_pos, mws, yerr=sds, color=colors, alpha=0.75, width=0.5,
           edgecolor="black", linewidth=0.8,
           error_kw=dict(elinewidth=1.2, capsize=5, capthick=1.2, ecolor="black"))

    ax.axhline(mean_sil, color="black", linewidth=1.0, linestyle="--",
               label=f"Gesamtmittelwert = {mean_sil:.2f}")
    ax.axhline(0.2, color="gray", linewidth=0.8, linestyle=":",
               label="Grenzwert s = 0.2")

    ax.set_xticks(list(x_pos))
    ax.set_xticklabels([str(cid) if str(cid).startswith("Cluster") else f"Cluster {cid}" for cid in cluster_ids], fontsize=10)
    ax.set_ylabel("Silhouette-Koeffizient [–]")
    ax.set_ylim(0, 0.8)
    ax.legend(frameon=True, fontsize=10)
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)

    fig.tight_layout()
    out_path = out_dir / f"silhouette_plot{suffix}.png"
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Plot gespeichert: {out_path}")


# ── Fatigue Feature-Matrix laden ─────────────────────────────────────────────

def _load_fatigue_feature_matrix(df_labels: pd.DataFrame) -> pd.DataFrame | None:
    """Laedt die z-transformierten Fatigue-Features (Delta/Slope) fuer Silhouette."""
    fatigue_csv = config.FATIGUE_FEATURES_CSV
    if not fatigue_csv.exists():
        return None

    df_fat = pd.read_csv(fatigue_csv)
    feature_cols = ["Delta_DF", "Slope_DF", "Delta_SF", "Slope_SF"]
    feature_cols = [c for c in feature_cols if c in df_fat.columns]
    if not feature_cols:
        return None

    subjects = df_labels["Subject"].tolist()
    df_fat = df_fat[df_fat["Subject"].isin(subjects)].copy()

    from sklearn.preprocessing import StandardScaler
    X_z = StandardScaler().fit_transform(df_fat[feature_cols].values)
    df_z = pd.DataFrame(X_z, columns=feature_cols)
    df_z.insert(0, "Subject", df_fat["Subject"].values)
    print(f"  Feature-Matrix: {feature_cols} (Fatigue-Features, z-transformiert)")
    return df_z


# ── Hilfsfunktion: einen Datensatz auswerten ──────────────────────────────────

def _run_for_labels(
    labels_path: Path,
    out_dir: Path,
    prefix: str,
    is_fatigue: bool = False,
) -> None:
    print(f"\nLese Cluster-Labels: {labels_path.name}")
    df_labels = pd.read_csv(labels_path)

    if "cluster_label" not in df_labels.columns:
        cols = [c for c in df_labels.columns if c != "Subject"]
        df_labels = df_labels.rename(columns={cols[0]: "cluster_label"})

    stem_suffix = labels_path.stem.replace("fatigue_cluster_labels", "").replace("cluster_labels", "")
    suffix = f"_{prefix}{stem_suffix}"

    if is_fatigue:
        df_z = _load_fatigue_feature_matrix(df_labels)
    else:
        df_z = _load_feature_matrix(df_labels)

    if df_z is None:
        print("FEHLER: Feature-Matrix konnte nicht geladen werden.")
        return

    silhouette_plot(df_z, df_labels, suffix, out_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Silhouette-Analyse ===\n")

    out_dir = config.OUTPUT_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stil-Clustering ───────────────────────────────────────────────────────
    print("--- Stil-Clustering ---")
    stil_path = _find_latest("cluster_labels*.csv")
    if stil_path is None:
        print("WARNUNG: Keine cluster_labels*.csv gefunden — Stil-Silhouette wird uebersprungen.")
        print("         Bitte zuerst main.py (Stil-Clustering) ausfuehren.")
    else:
        _run_for_labels(stil_path, out_dir, prefix="stil", is_fatigue=False)

    # ── Fatigue-Clustering ────────────────────────────────────────────────────
    print("\n--- Fatigue-Clustering ---")
    fatigue_path = _find_latest("fatigue_cluster_labels*.csv")
    if fatigue_path is None:
        print("WARNUNG: Keine fatigue_cluster_labels*.csv gefunden — Fatigue-Silhouette wird uebersprungen.")
        print("         Bitte zuerst main.py (Fatigue-Clustering) ausfuehren.")
    else:
        _run_for_labels(fatigue_path, out_dir, prefix="fatigue", is_fatigue=True)

    print("\n=== Fertig ===")


if __name__ == "__main__":
    main()
