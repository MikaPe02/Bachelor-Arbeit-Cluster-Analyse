# exploration/22_verlaufsplot.py
#
# ZWECK: Verlaufsplot von DF und SF_norm ueber alle km-Marken pro Cluster.
#        Mittelwert + ±1 SD Band pro Cluster, eingefaerbt nach Paul-Tol-Palette.
#        Liest cluster_labels.csv und dual_axis_dataset.csv — kein Pipeline-Lauf noetig.
#
# AUFRUF:
#   python exploration/22_verlaufsplot.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import config

# ── Konstanten ────────────────────────────────────────────────────────────────

_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

_RCPARAMS = {
    "font.size":        11,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
}

FIGSIZE   = (16 / 2.54, 13 / 2.54)
DPI       = 300


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _find_latest(pattern: str) -> Path | None:
    files = sorted(
        config.OUTPUT_DATA_DIR.glob(pattern),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _cluster_label_col(df_labels: pd.DataFrame) -> str:
    for col in ("cluster_label", "fatigue_cluster_label"):
        if col in df_labels.columns:
            return col
    raise ValueError("Keine Cluster-Label-Spalte gefunden in cluster_labels.csv")


# ── Plot ──────────────────────────────────────────────────────────────────────

def verlaufsplot(
    df_long: pd.DataFrame,
    df_labels: pd.DataFrame,
    feature: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """
    Zeichnet Mittelwert ± 1 SD pro Cluster ueber alle km-Marken.

    Parameters
    ----------
    df_long  : langer Datensatz mit Spalten Subject, km, <feature>
    df_labels: DataFrame mit Spalten Subject, cluster_label
    feature  : Spaltenname der abzubildenden Variable ('DF' oder 'SF_norm')
    ylabel   : Achsenbeschriftung
    out_path : Speicherpfad
    """
    label_col = _cluster_label_col(df_labels)
    df = df_long.merge(df_labels[["Subject", label_col]], on="Subject", how="inner")
    df = df.rename(columns={label_col: "cluster_label"})

    cluster_ids = sorted(df["cluster_label"].dropna().unique(), key=str)
    km_marks    = sorted(df["km"].unique())

    plt.rcParams.update(_RCPARAMS)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    for i, cid in enumerate(cluster_ids):
        color  = _PALETTE[i % len(_PALETTE)]
        sub    = df[df["cluster_label"] == cid]

        means = []
        sds   = []
        kms   = []

        for km in km_marks:
            vals = sub.loc[np.abs(sub["km"] - km) <= 1e-6, feature].dropna()
            if len(vals) < 2:
                continue
            means.append(vals.mean())
            sds.append(vals.std(ddof=1))
            kms.append(km)

        means = np.array(means)
        sds   = np.array(sds)
        kms   = np.array(kms)

        lbl = str(cid) if str(cid).startswith("Cluster") else f"Cluster {cid}"
        ax.plot(kms, means, color=color, linewidth=1.8, label=lbl)

    ax.set_xlabel("Strecke [km]")
    ax.set_ylabel(ylabel)
    ax.set_xticks(km_marks)
    ax.grid(True, linewidth=0.5, alpha=0.5)
    for spine in ax.spines.values():
        spine.set_visible(True)

    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Gespeichert: {out_path}")


# ── Hilfsfunktion: einen Datensatz plotten ────────────────────────────────────

def _run_for_labels(df_long: pd.DataFrame, labels_path: Path, out_dir: Path, prefix: str) -> None:
    print(f"\nLese Cluster-Labels: {labels_path.name}")
    df_labels = pd.read_csv(labels_path)

    # Suffix = Typ-Praefix + eventueller Methoden-Suffix aus dem Dateinamen
    stem_suffix = labels_path.stem.replace("fatigue_cluster_labels", "").replace("cluster_labels", "")
    suffix = f"_{prefix}{stem_suffix}"

    verlaufsplot(
        df_long, df_labels,
        feature="DF",
        ylabel="Duty Factor [–]",
        out_path=out_dir / f"verlaufsplot_DF{suffix}.png",
    )
    verlaufsplot(
        df_long, df_labels,
        feature="SF_norm",
        ylabel="Normierte Schrittfrequenz [–]",
        out_path=out_dir / f"verlaufsplot_SF_norm{suffix}.png",
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Verlaufsplot ===\n")

    # Daten laden
    if not config.DUAL_AXIS_CSV.exists():
        print(f"FEHLER: {config.DUAL_AXIS_CSV} nicht gefunden.")
        print("Bitte zuerst extract_to_csv.py ausfuehren.")
        return

    df_long = pd.read_csv(config.DUAL_AXIS_CSV)

    if "speed_ms" not in df_long.columns and config.SUBJECTS_CSV.exists():
        subj = pd.read_csv(config.SUBJECTS_CSV, usecols=["Subject", "speed_ms"])
        df_long = df_long.merge(subj, on="Subject", how="left")

    out_dir = config.OUTPUT_PLOTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Stil-Clustering ───────────────────────────────────────────────────────
    print("--- Stil-Clustering ---")
    stil_path = _find_latest("cluster_labels*.csv")
    if stil_path is None:
        print("WARNUNG: Keine cluster_labels*.csv gefunden — Stil-Verlaufsplot wird uebersprungen.")
        print("         Bitte zuerst main.py (Stil-Clustering) ausfuehren.")
    else:
        _run_for_labels(df_long, stil_path, out_dir, prefix="stil")

    # ── Fatigue-Clustering ────────────────────────────────────────────────────
    print("\n--- Fatigue-Clustering ---")
    fatigue_path = _find_latest("fatigue_cluster_labels*.csv")
    if fatigue_path is None:
        print("WARNUNG: Keine fatigue_cluster_labels*.csv gefunden — Fatigue-Verlaufsplot wird uebersprungen.")
        print("         Bitte zuerst main.py (Fatigue-Clustering) ausfuehren.")
    else:
        _run_for_labels(df_long, fatigue_path, out_dir, prefix="fatigue")

    print("\n=== Fertig ===")


if __name__ == "__main__":
    main()
