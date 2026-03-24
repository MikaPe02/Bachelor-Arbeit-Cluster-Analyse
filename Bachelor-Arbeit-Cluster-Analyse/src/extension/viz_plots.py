# src/extension/viz_plots.py
#
# ZWECK: Publikationsfertige Dual-Axis Visualisierungen.
#
# FUNKTIONEN:
#   dual_axis_snapshot() – Ausgangslaufstile bei km 1.0, keine Clustereinfaerbung
#   dual_axis_arrows()   – Verlaufspfeile km 1.0 → 9.5, eingefaerbt nach Cluster
#   cluster_scatter()    – Clustering-Ergebnis bei km 1.0 mit Zentroiden
#
# VERWENDUNG:
#   from extension.viz_plots import dual_axis_snapshot, dual_axis_arrows, cluster_scatter

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Konstanten ────────────────────────────────────────────────────────────────

# Positionen der van Oeveren Regionen im Dual-Axis Raum
# Quelle: van Oeveren et al. (2021)
_REGION_LABELS: list[tuple[str, float, float]] = [
    ("Stick",  0.71, 0.76),
    ("Bounce", 0.63, 0.86),
    ("Push",   0.55, 0.96),
    ("Hop",    0.53, 1.00),
    ("Sit",    0.69, 0.74),
]

# Farb-Palette (farbenblinden-freundlich, Paul Tol)
_PALETTE = [
    "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE",
    "#AA3377", "#BBBBBB",
]

_BASE_RCPARAMS = {
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}

FIGSIZE = (8.0, 6.5)
DPI     = 300
XLIM    = (0.45, 0.78)   # DF-Achse, einheitlich fuer alle drei Plots
YLIM    = (0.65, 1.08)   # SF_norm-Achse, einheitlich fuer alle drei Plots


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _snapshot_at(df: pd.DataFrame, km: float, tol: float = 1e-6) -> pd.DataFrame:
    """Gibt alle Zeilen fuer eine km-Marke zurueck."""
    return df[np.abs(df["km"] - km) <= tol].copy()


def _build_color_map(labels: pd.Series) -> dict:
    """Stabile Farb-Zuordnung: sortierte Labels -> Farben aus _PALETTE."""
    unique = sorted(labels.unique(), key=str)
    return {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(unique)}


def _save(fig: plt.Figure, out_dir: Optional[Path], filename: str) -> None:
    if out_dir is None:
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, dpi=DPI, bbox_inches="tight")


def _add_region_labels(ax: plt.Axes) -> None:
    """Zeichnet die 5 van Oeveren Regionen als Hintergrundbeschriftung."""
    for name, df_pos, sf_pos in _REGION_LABELS:
        ax.text(
            df_pos, sf_pos, name,
            fontsize=10, fontstyle="italic",
            color="gray", alpha=0.5,
            ha="center", va="center",
            zorder=0,
        )


def _set_axes(ax: plt.Axes) -> None:
    ax.set_xlabel("Duty Factor (DF)")
    ax.set_ylabel("SF norm")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.grid(True, linewidth=0.4, alpha=0.6)


# ── Plot 1 ────────────────────────────────────────────────────────────────────

def dual_axis_snapshot(
    df: pd.DataFrame,
    *,
    out_dir: Optional[Path] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 1: Ausgangslaufstile aller Probanden bei km=1.0.

    Keine Clustereinfaerbung – alle Punkte einheitlich.
    Beschriftung mit Subject-ID, Hintergrundregionen nach van Oeveren.

    Parameters
    ----------
    df      : langer Datensatz (Subject x km), muss DF, SF_norm, Subject enthalten
    out_dir : Speicherordner; None = nicht speichern
    """
    plt.rcParams.update(_BASE_RCPARAMS)

    snap = _snapshot_at(df, km=1.0)
    if snap.empty:
        raise ValueError("Keine Daten bei km=1.0 gefunden.")

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("Dual-Axis Framework \u2013 Ausgangslaufstile bei km 1.0")
    _set_axes(ax)
    _add_region_labels(ax)

    ax.scatter(
        snap["DF"], snap["SF_norm"],
        color="#4477AA", s=60, zorder=3, alpha=0.85,
    )

    # Subject-Labels leicht versetzt
    for _, row in snap.iterrows():
        ax.annotate(
            row["Subject"],
            xy=(row["DF"], row["SF_norm"]),
            xytext=(3, 3), textcoords="offset points",
            fontsize=7, color="#333333",
        )

    fig.tight_layout()
    _save(fig, out_dir, "dual_axis_snapshot.png")
    return fig, ax


# ── Plot 2 ────────────────────────────────────────────────────────────────────

def dual_axis_arrows(
    df: pd.DataFrame,
    *,
    cluster_col: str = "true_running_style",
    out_dir: Optional[Path] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 2: Verlaufspfeile von km=1.0 zu km=9.5, eingefaerbt nach Cluster.

    Parameters
    ----------
    df          : langer Datensatz mit Spalte cluster_col
    cluster_col : Spaltenname fuer Cluster-Zugehoerigkeit
    out_dir     : Speicherordner; None = nicht speichern
    """
    plt.rcParams.update(_BASE_RCPARAMS)

    start = _snapshot_at(df, km=1.0).rename(
        columns={"DF": "DF_start", "SF_norm": "SF_start"}
    )
    end   = _snapshot_at(df, km=9.5).rename(
        columns={"DF": "DF_end", "SF_norm": "SF_end"}
    )

    if start.empty or end.empty:
        raise ValueError("Keine Daten bei km=1.0 oder km=9.5 gefunden.")

    # Cluster-Spalte aus Start-Snapshot uebernehmen
    keep = ["Subject", "DF_start", "SF_start", cluster_col]
    merged = start[keep].merge(
        end[["Subject", "DF_end", "SF_end"]],
        on="Subject", how="inner",
    )

    color_map = _build_color_map(merged[cluster_col])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("Erm\u00fcdungsverlauf im Dual-Axis Raum (km 1.0 \u2192 km 9.5)")
    _set_axes(ax)
    _add_region_labels(ax)

    # Startpunkte als Kreise
    for lab in sorted(color_map):
        sub = merged[merged[cluster_col] == lab]
        c   = color_map[lab]

        ax.scatter(sub["DF_start"], sub["SF_start"],
                   color=c, s=40, zorder=3, alpha=0.7)

        for _, row in sub.iterrows():
            ax.annotate(
                "",
                xy     = (row["DF_end"],   row["SF_end"]),
                xytext = (row["DF_start"], row["SF_start"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=c,
                    lw=1.6,
                    alpha=0.85,
                ),
                zorder=4,
            )

    # Legende
    handles = [
        mpatches.Patch(color=color_map[lab], label=str(lab))
        for lab in sorted(color_map)
    ]
    ax.legend(handles=handles, title=cluster_col, loc="best", frameon=True)

    fig.tight_layout()
    _save(fig, out_dir, "dual_axis_arrows.png")
    return fig, ax


# ── Plot 3 ────────────────────────────────────────────────────────────────────

def cluster_scatter(
    df: pd.DataFrame,
    *,
    cluster_col: str = "true_running_style",
    out_dir: Optional[Path] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot 3: Clustering-Ergebnis bei km=1.0 mit Zentroiden.

    Parameters
    ----------
    df          : langer Datensatz mit Spalte cluster_col
    cluster_col : Spaltenname fuer Cluster-Zugehoerigkeit
    out_dir     : Speicherordner; None = nicht speichern
    """
    plt.rcParams.update(_BASE_RCPARAMS)

    snap = _snapshot_at(df, km=1.0)
    if snap.empty:
        raise ValueError("Keine Daten bei km=1.0 gefunden.")

    color_map = _build_color_map(snap[cluster_col])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_title("Clustering-Ergebnis im Dual-Axis Raum")
    _set_axes(ax)
    _add_region_labels(ax)

    for lab in sorted(color_map):
        sub = snap[snap[cluster_col] == lab]
        c   = color_map[lab]

        ax.scatter(sub["DF"], sub["SF_norm"],
                   color=c, s=60, alpha=0.85, zorder=3, label=str(lab))

        # Zentroid als X
        cx = sub["DF"].mean()
        cy = sub["SF_norm"].mean()
        ax.scatter(cx, cy,
                   marker="X", s=180, color=c,
                   edgecolors="black", linewidths=0.8,
                   zorder=5)

    ax.legend(title=cluster_col, loc="best", frameon=True)

    fig.tight_layout()
    _save(fig, out_dir, "cluster_scatter.png")
    return fig, ax


# ── Plot 4 ────────────────────────────────────────────────────────────────────

def metrics_table(
    df_results: pd.DataFrame,
    *,
    out_dir: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot 4: Wissenschaftliche Tabelle der Clustering-Validierungsmetriken.

    Booktabs-Stil: nur horizontale Linien, kein Farbe, schwarz/weiss/grau.
    Pro Methode/Linkage-Gruppe: best_k − 1, best_k, best_k + 1.
    Beste Zeile hellgrau hinterlegt, Werte fett.
    k ausserhalb des getesteten Bereichs: grauer Text, Werte als '—'.

    Parameters
    ----------
    df_results : Ausgabe von evaluate_clustering_methods() + add_best_flag()
    out_dir    : Speicherordner; None = nicht speichern
    """
    plt.rcParams.update(_BASE_RCPARAMS)

    # ── Spalten: Beschriftung und Mittelpunkte (normiert 0–1) ─────────────────
    COL_LABELS  = ["Methode", "Linkage", "k",
                   "Silhouette \u2191", "Davies-Bouldin \u2193",
                   "Calinski-Harabasz \u2191"]
    # Breiten: 0.175 + 0.115 + 0.065 + 0.205 + 0.215 + 0.225 = 1.0
    COL_CENTERS = [0.088, 0.233, 0.323, 0.460, 0.670, 0.888]

    # ── Gruppen aufbauen ──────────────────────────────────────────────────────
    groups = []
    for linkage in ["ward", "complete", "average", "single"]:
        mask = (df_results["method"] == "hierarchical") & (df_results["linkage"] == linkage)
        if mask.any():
            groups.append(("hierarchical", linkage, df_results[mask].copy()))
    mask = df_results["method"] == "kmeans"
    if mask.any():
        groups.append(("kmeans", None, df_results[mask].copy()))

    # ── Zeilen aufbauen ───────────────────────────────────────────────────────
    display_rows: list[dict] = []

    for method, linkage, df_grp in groups:
        best_rows = df_grp[df_grp["is_best"] == True]
        if best_rows.empty:
            continue
        best_k      = int(best_rows.iloc[0]["k"])
        method_lbl  = "Hierarchisch" if method == "hierarchical" else "k-Means"
        linkage_lbl = linkage.capitalize() if linkage else "\u2014"

        for k_val in [best_k - 1, best_k, best_k + 1]:
            match  = df_grp[df_grp["k"] == k_val]
            exists  = not match.empty
            is_best = (k_val == best_k)

            if exists:
                r = match.iloc[0]
                row_data = [
                    method_lbl, linkage_lbl, str(k_val),
                    f"{r['silhouette']:.4f}",
                    f"{r['davies_bouldin']:.4f}",
                    f"{r['calinski_harabasz']:.2f}",
                ]
            else:
                row_data = [method_lbl, linkage_lbl, str(k_val),
                            "\u2014", "\u2014", "\u2014"]

            display_rows.append({
                "data":          row_data,
                "is_best":       is_best,
                "exists":        exists,
                "last_in_group": False,
            })

        display_rows[-1]["last_in_group"] = True

    # ── Figur-Groesse ─────────────────────────────────────────────────────────
    n_data       = len(display_rows)
    n_total_rows = n_data + 1          # +1 Header
    row_h_in     = 0.30
    table_h_in   = n_total_rows * row_h_in
    title_h_in   = 0.50
    sep_h_in     = 0.18
    legend_h_in  = 1.55
    pad_in       = 0.28
    fig_h        = title_h_in + table_h_in + sep_h_in + legend_h_in + pad_in

    fig = plt.figure(figsize=(12.0, fig_h))

    # Normierte y-Positionen (Ursprung unten)
    legend_bot  = (pad_in / 2) / fig_h
    legend_frac = legend_h_in / fig_h
    sep_frac    = sep_h_in / fig_h
    table_bot   = legend_bot + legend_frac + sep_frac
    table_frac  = table_h_in / fig_h
    title_bot   = table_bot + table_frac

    # ── Titel ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, title_bot + (title_h_in / fig_h) * 0.45,
        "Vergleich der Clustering-Methoden \u2013 Validierungsmetriken",
        ha="center", va="center", fontsize=12, fontweight="bold",
    )

    # ── Tabellen-Achse ────────────────────────────────────────────────────────
    ax = fig.add_axes([0.01, table_bot, 0.98, table_frac])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    row_h = 1.0 / n_total_rows          # Zeilenhoehe in Achsen-Koordinaten

    # y-Positionen: Header ganz oben
    header_bot = 1.0 - row_h
    header_ctr = 1.0 - row_h / 2.0
    data_ctrs  = [1.0 - row_h * (i + 1.5) for i in range(n_data)]
    data_bots  = [1.0 - row_h * (i + 2.0) for i in range(n_data)]

    # ── Hintergruende ─────────────────────────────────────────────────────────
    ax.add_patch(mpatches.Rectangle(
        (0, header_bot), 1.0, row_h,
        facecolor="#2D2D2D", edgecolor="none", zorder=1, clip_on=False,
        transform=ax.transData,
    ))

    for i, meta in enumerate(display_rows):
        if meta["is_best"]:
            ax.add_patch(mpatches.Rectangle(
                (0, data_bots[i]), 1.0, row_h,
                facecolor="#F0F0F0", edgecolor="none", zorder=1, clip_on=False,
                transform=ax.transData,
            ))

    # ── Text: Header ─────────────────────────────────────────────────────────
    for j, lbl in enumerate(COL_LABELS):
        ax.text(COL_CENTERS[j], header_ctr, lbl,
                ha="center", va="center",
                fontsize=10, fontweight="bold", color="white",
                zorder=3)

    # ── Text: Datenzeilen ─────────────────────────────────────────────────────
    for i, meta in enumerate(display_rows):
        txt_color = "#999999" if not meta["exists"] else "black"
        weight    = "bold"    if meta["is_best"]    else "normal"
        for j, val in enumerate(meta["data"]):
            ax.text(COL_CENTERS[j], data_ctrs[i], val,
                    ha="center", va="center",
                    fontsize=10, color=txt_color, fontweight=weight,
                    zorder=3)

    # ── Horizontale Linien (booktabs) ─────────────────────────────────────────
    def _hline(y: float, lw: float, color: str = "black") -> None:
        ax.plot([0, 1], [y, y], color=color, linewidth=lw,
                transform=ax.transData, zorder=4,
                clip_on=False, solid_capstyle="butt")

    _hline(header_bot, lw=1.5)           # midrule

    for i, meta in enumerate(display_rows):
        y = data_bots[i]
        if i == n_data - 1:
            _hline(y, lw=2.0)            # bottomrule
        elif meta["last_in_group"]:
            _hline(y, lw=2.0)            # Gruppen-Trennlinie
        else:
            _hline(y, lw=0.5, color="#888888")  # Zeilen-Trennlinie

    # ── Trennlinie Tabelle / Legende ──────────────────────────────────────────
    sep_y = table_bot - 0.004
    fig.add_artist(mlines.Line2D(
        [0.01, 0.99], [sep_y, sep_y],
        transform=fig.transFigure,
        color="#888888", linewidth=0.5, zorder=5,
    ))

    # ── Legende ───────────────────────────────────────────────────────────────
    ax_leg = fig.add_axes([0.01, legend_bot, 0.98, legend_frac - 0.01])
    ax_leg.axis("off")

    legend_text = (
        "Kennwerte zur Bewertung der Clustering-Qualit\u00e4t"
        " (Quelle: Arbelaitz et al., 2013, Pattern Recognition):\n\n"

        "Silhouette-Koeffizient \u2191\u2002\u2002"
        "Bereich: \u22121 bis +1.\u2002"
        "Misst Koh\u00e4sion und Separation der Cluster.\u2002"
        "Werte nahe +1\u202f=\u202fgut getrennte, kompakte Cluster.\u2002"
        "Werte nahe 0\u202f=\u202f\u00fcberlappend.\u2002"
        "Werte\u202f<\u202f0\u202f=\u202ffehlerhafte Zuweisung.\n\n"

        "Davies-Bouldin-Index \u2193\u2002\u2002\u2002\u2002"
        "Bereich: \u2265\u20020.\u2002"
        "Verh\u00e4ltnis aus mittlerer Streuung innerhalb der Cluster"
        " zum Abstand zwischen den Clusterzentren.\u2002"
        "Niedrigere Werte\u202f=\u202fbessere Trennung.\u2002"
        "0\u202f=\u202ftheoretisches Optimum.\n\n"

        "Calinski-Harabasz-Index \u2191\u2002"
        "Bereich: \u2265\u20020.\u2002"
        "Verh\u00e4ltnis der Zwischen-Cluster-Varianz zur Binnen-Cluster-Varianz.\u2002"
        "H\u00f6here Werte\u202f=\u202fkompaktere, besser getrennte Cluster.\u2002"
        "Kein absolutes Optimum \u2013 nur im Vergleich interpretierbar.\n\n"

        "Grau hinterlegt\u202f=\u202fbeste Konfiguration je Methode/Linkage"
        " (Rang-Aggregation \u00fcber alle drei Kennwerte).\u2003"
        "Grauer Text\u202f=\u202fk nicht im getesteten Bereich.\u2003"
        "\u2191\u202fh\u00f6her ist besser.\u2003"
        "\u2193\u202fniedriger ist besser."
    )

    ax_leg.text(
        0.0, 1.0, legend_text,
        transform=ax_leg.transAxes,
        va="top", ha="left",
        fontsize=8, linespacing=1.55,
        color="#222222",
    )

    _save(fig, out_dir, "metrics_table.png")
    plt.close(fig)
    return fig


# ── Plot 5 ────────────────────────────────────────────────────────────────────

def elbow_plot(
    df_results: pd.DataFrame,
    df_features_z: pd.DataFrame,
    cfg,
) -> None:
    """
    Plot 5: Elbow-Plot – normierte Innerhalb-Cluster-Streuung gegen k.

    Inertia (Within-Cluster Sum of Squares, WCSS) misst die Gesamtstreuung
    aller Punkte innerhalb ihrer jeweiligen Cluster.
    Bei k=1 liegt alles in einem Cluster → maximale Streuung = 100 %.
    Mit jedem zusaetzlichen Cluster sinkt die Streuung.
    Der 'Knick' (Elbow) markiert den Punkt, ab dem mehr Cluster
    kaum noch zusaetzliche Erklaerungskraft liefern.

    Parameters
    ----------
    df_results    : Ausgabe von evaluate_clustering_methods() + add_best_flag()
    df_features_z : Z-transformierte Feature-Tabelle mit Spalte 'Subject'
    cfg           : config-Modul (benoetigt RANDOM_STATE, OUTPUT_PLOTS_DIR)
    """
    from sklearn.cluster import KMeans
    import matplotlib.ticker as mticker

    print("\n=== Schritt 4b: Elbow-Plot ===")

    df_km = df_results[df_results["method"] == "kmeans"].copy()
    if df_km.empty:
        print("  Uebersprungen: keine k-Means Ergebnisse (RUN_KMEANS=False).")
        return

    X = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)
    k1_inertia = float(
        KMeans(n_clusters=1, random_state=cfg.RANDOM_STATE, n_init=10)
        .fit(X).inertia_
    )

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

    cfg.OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_PLOTS_DIR / "elbow_plot.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Elbow-Plot gespeichert -> {out}")


# ── Alle Plots auf einmal ─────────────────────────────────────────────────────

def create_all_plots(
    df: pd.DataFrame,
    labels: "pd.Series",
    df_results: pd.DataFrame,
    cfg,
) -> None:
    """
    Erzeugt alle vier Dual-Axis Plots mit den finalen Cluster-Labels.

    Parameters
    ----------
    df         : langer Datensatz (Subject x km) ohne cluster_label
    labels     : pd.Series mit Index=Subject, Values=Cluster-Label
    df_results : Ausgabe von evaluate_clustering_methods() + add_best_flag()
    cfg        : config-Modul (benoetigt OUTPUT_PLOTS_DIR)
    """
    print("\n=== Schritt 5b: Plots erstellen ===")

    cfg.OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df_labels  = labels.reset_index()
    df_labels.columns = ["Subject", "cluster_label"]
    df_labeled = df.merge(df_labels, on="Subject", how="left")

    print("  Plot 1: dual_axis_snapshot ...")
    dual_axis_snapshot(df_labeled, out_dir=cfg.OUTPUT_PLOTS_DIR)

    print("  Plot 2: dual_axis_arrows ...")
    dual_axis_arrows(df_labeled, cluster_col="cluster_label", out_dir=cfg.OUTPUT_PLOTS_DIR)

    print("  Plot 3: cluster_scatter ...")
    cluster_scatter(df_labeled, cluster_col="cluster_label", out_dir=cfg.OUTPUT_PLOTS_DIR)

    print("  Plot 4: metrics_table ...")
    metrics_table(df_results, out_dir=cfg.OUTPUT_PLOTS_DIR)

    plt.close("all")
    print(f"  Plots gespeichert -> {cfg.OUTPUT_PLOTS_DIR}")
