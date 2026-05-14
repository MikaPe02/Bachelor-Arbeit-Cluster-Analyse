# exploration/02_cluster_speed_independence.py
#
# ZIEL: Prüfen ob Cluster durch Speed erklärt werden (Post-hoc Validierung).
#       ANOVA: Sind Speed-Mittelwerte zwischen Clustern signifikant verschieden?
#
# VORAUSSETZUNG:
#   - Outputs/Data/cluster_labels.csv  (erzeugt von main.py)
#   - Input/subjects.csv                (speed_ms pro Proband)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

CLUSTER_LABELS_CSV   = PROJECT_ROOT / "Outputs" / "Data" / "cluster_labels.csv"
CLUSTER_RESULTS_CSV  = PROJECT_ROOT / "Outputs" / "Data" / "cluster_results.csv"
SUBJECTS_CSV         = PROJECT_ROOT / "Input" / "subjects.csv"
OUTPUT_PLOTS         = PROJECT_ROOT / "Outputs" / "Plots"
OUTPUT_DATA          = PROJECT_ROOT / "Outputs" / "Data"


def load_data() -> pd.DataFrame:
    for path, name in [
        (CLUSTER_LABELS_CSV, "cluster_labels.csv"),
        (SUBJECTS_CSV,       "subjects.csv"),
    ]:
        if not path.exists():
            raise SystemExit(
                f"FEHLER: {name} nicht gefunden: {path}\n"
                + ("Bitte zuerst main.py ausfuehren." if "cluster" in name
                   else "Bitte zuerst extract_to_csv.py ausfuehren.")
            )

    labels   = pd.read_csv(CLUSTER_LABELS_CSV)
    subjects = pd.read_csv(SUBJECTS_CSV, usecols=["Subject", "speed_ms"])

    df = labels.merge(subjects, on="Subject", how="left")
    df["speed_kmh"] = df["speed_ms"] * 3.6
    df["cluster_label"] = df["cluster_label"].astype(str)

    missing = df["speed_ms"].isna().sum()
    if missing > 0:
        print(f"  WARNUNG: {missing} Probanden ohne Speed-Wert werden ausgeschlossen.")
        df = df.dropna(subset=["speed_ms"])

    return df



def _load_method_label() -> str:
    """Liest das gewaelte Clusterverfahren aus cluster_results.csv."""
    if not CLUSTER_RESULTS_CSV.exists():
        return ""
    try:
        df_res = pd.read_csv(CLUSTER_RESULTS_CSV)
        best = df_res[df_res["is_best"] == True]
        if best.empty:
            best = df_res
        row = best.iloc[0]
        method = str(row.get("method", ""))
        if method == "kmeans":
            return f"k-Means, k={int(row['k'])}"
        if method == "hierarchical":
            linkage = str(row.get("linkage", "")).capitalize()
            return f"Hierarchisch ({linkage}), k={int(row['k'])}"
        if method == "hdbscan":
            mcs = row.get("min_cluster_size", "?")
            return f"HDBSCAN, mcs={int(mcs)}"
        return method
    except Exception:
        return ""


def run_anova(df: pd.DataFrame) -> tuple[float, float]:
    groups = [grp["speed_kmh"].values for _, grp in df.groupby("cluster_label")]
    f_stat, p_val = stats.f_oneway(*groups)
    return float(f_stat), float(p_val)


def make_plots(df: pd.DataFrame, f_stat: float, p_val: float, method_label: str = "") -> Path:
    sns.set_style("whitegrid")
    cluster_ids = sorted(df["cluster_label"].unique())
    n_clusters  = len(cluster_ids)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Signifikanz-Label
    if p_val < 0.001:   sig = "p < 0.001 ***"
    elif p_val < 0.01:  sig = f"p = {p_val:.3f} **"
    elif p_val < 0.05:  sig = f"p = {p_val:.3f} *"
    else:               sig = f"p = {p_val:.3f} ns"

    entscheidung = (
        "Cluster sind Speed-abhaengig!"
        if p_val < 0.05
        else "Cluster sind Speed-unabhaengig"
    )
    method_str = f" | Verfahren: {method_label}" if method_label else ""
    fig.suptitle(
        f"Speed-Unabhaengigkeit der Cluster{method_str}\n"
        f"ANOVA: F = {f_stat:.2f}, {sig}  ->  {entscheidung}",
        fontsize=12, fontweight="bold",
    )

    palette = sns.color_palette("tab10", n_clusters)

    # Plot 1: Boxplot
    ax = axes[0]
    sns.boxplot(
        data=df, x="cluster_label", y="speed_kmh",
        order=cluster_ids, palette=palette, ax=ax,
        width=0.5, linewidth=1.5,
    )
    sns.stripplot(
        data=df, x="cluster_label", y="speed_kmh",
        order=cluster_ids, palette=palette, ax=ax,
        size=5, alpha=0.6, jitter=True,
    )
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Speed [km/h]", fontsize=11)
    ax.set_title("Speed-Verteilung pro Cluster", fontsize=11)

    # Plot 2: Barplot (Mean ± SD)
    ax = axes[1]
    means = df.groupby("cluster_label")["speed_kmh"].mean().reindex(cluster_ids)
    sds   = df.groupby("cluster_label")["speed_kmh"].std().reindex(cluster_ids)
    x_pos = np.arange(n_clusters)

    bars = ax.bar(x_pos, means, yerr=sds, color=palette,
                  capsize=5, width=0.5, alpha=0.85, linewidth=1.2,
                  error_kw={"linewidth": 1.5})
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_ids)
    ax.set_xlabel("Cluster", fontsize=11)
    ax.set_ylabel("Speed [km/h]", fontsize=11)
    ax.set_title("Mittlere Speed pro Cluster (Mean +/- SD)", fontsize=11)

    # N pro Cluster annotieren
    for i, cid in enumerate(cluster_ids):
        n = len(df[df["cluster_label"] == cid])
        ax.text(i, 0.5, f"n={n}", ha="center", va="bottom", fontsize=9, color="black")

    plt.tight_layout()
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_PLOTS / "cluster_speed_independence.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    print("=" * 65)
    print("CLUSTER-SPEED-UNABHAENGIGKEIT  (ANOVA)")
    print("=" * 65)

    df = load_data()

    cluster_ids = sorted(df["cluster_label"].unique())
    print(f"\n  Cluster:  {cluster_ids}")
    print(f"  Probanden gesamt: {len(df)}")

    print("\n  Speed [km/h] pro Cluster:")
    print(f"  {'Cluster':<10} {'N':>4}  {'Mean':>6}  {'SD':>6}  {'Min':>6}  {'Max':>6}")
    print("  " + "-" * 46)
    for cid in cluster_ids:
        grp = df[df["cluster_label"] == cid]["speed_kmh"]
        print(f"  {cid:<10} {len(grp):>4}  {grp.mean():>6.2f}  "
              f"{grp.std():>6.2f}  {grp.min():>6.2f}  {grp.max():>6.2f}")

    f_stat, p_val = run_anova(df)

    if p_val < 0.001:   sig = "***"
    elif p_val < 0.01:  sig = "**"
    elif p_val < 0.05:  sig = "*"
    else:               sig = "ns"

    print(f"\n  ANOVA:  F = {f_stat:.3f},  p = {p_val:.4f}  {sig}")
    print()
    print("=" * 65)
    if p_val < 0.05:
        print(f"  ERGEBNIS: Cluster sind SPEED-ABHAENGIG (p = {p_val:.4f})")
        print("  -> Cluster unterscheiden sich signifikant in der Speed.")
        print("     Bei Clustering auf Rohdaten erwartet (Speed-Gruppen).")
        print("     Bei Residual-Clustering deutet das auf unvollstaendige")
        print("     Speed-Bereinigung hin.")
    else:
        print(f"  ERGEBNIS: Cluster sind SPEED-UNABHAENGIG (p = {p_val:.4f})")
        print("  -> Kein signifikanter Speed-Unterschied zwischen Clustern.")
        print("     Clustering spiegelt Laufstil, nicht Geschwindigkeit.")
    print("=" * 65)

    method_label = _load_method_label()
    plot_path = make_plots(df, f_stat, p_val, method_label)
    print(f"\n  Plot gespeichert: {plot_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
