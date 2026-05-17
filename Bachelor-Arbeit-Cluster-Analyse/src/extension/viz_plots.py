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
    ("Bounce", 0.53, 0.86),   # kleiner DF, mittlere SF_norm
    ("Hop",    0.62, 0.97),   # mittlerer DF, hohe SF_norm
    ("Sit",    0.62, 0.86),   # Mitte beides
    ("Push",   0.62, 0.77),   # mittlerer DF, kleine SF_norm
    ("Stick",  0.76, 0.86),   # grosser DF, mittlere SF_norm
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
XLIM    = (0.45, 0.88)   # DF-Achse, einheitlich fuer alle drei Plots
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



# ── Hilfsfunktion Methodenbeschriftung ───────────────────────────────────────

def _method_label(selection):
    """Kurze lesbare Beschriftung des gewaehlten Clustering-Verfahrens."""
    if selection is None:
        return ""
    method = selection.get("method", "")
    if method == "kmeans":
        label = f"k-Means, k={selection.get('k', '?')}"
    elif method == "hierarchical":
        linkage = selection.get("linkage", "").capitalize()
        label = f"Hierarchisch ({linkage}), k={selection.get('k', '?')}"
    elif method == "hdbscan":
        mcs = selection.get("min_cluster_size", "?")
        label = f"HDBSCAN, mcs={mcs}"
    else:
        label = method
    speed_str = "speedbereinigt" if selection.get("speed_corrected") else "Rohdaten"
    return f"{label} | {speed_str}"



def _file_suffix(selection):
    """Kurzer Dateiname-Suffix aus dem gewaehlten Verfahren, z.B. '_kmeans_k3_speedber'."""
    if selection is None:
        return ""
    method = selection.get("method", "")
    if method == "kmeans":
        base = f"_kmeans_k{selection.get('k', '?')}"
    elif method == "hierarchical":
        linkage = selection.get("linkage", "")
        base = f"_hierarchisch_{linkage}_k{selection.get('k', '?')}"
    elif method == "hdbscan":
        mcs = selection.get("min_cluster_size", "?")
        ms  = selection.get("min_samples", "auto")
        base = f"_hdbscan_mcs{mcs}_ms{ms}"
    else:
        base = f"_{method}"
    speed_str = "_speedber" if selection.get("speed_corrected") else "_roh"
    return base + speed_str


# ── Plot 1 ────────────────────────────────────────────────────────────────────

def dual_axis_snapshot(
    df: pd.DataFrame,
    *,
    selection=None,
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
    method_str = _method_label(selection)
    suffix = f" ({method_str})" if method_str else ""
    ax.set_title(f"Dual-Axis Framework \u2013 Ausgangslaufstile bei km 1.0{suffix}")
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
    selection=None,
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

    km_end = float(df["km"].max())

    start = _snapshot_at(df, km=1.0).rename(
        columns={"DF": "DF_start", "SF_norm": "SF_start"}
    )
    end   = _snapshot_at(df, km=km_end).rename(
        columns={"DF": "DF_end", "SF_norm": "SF_end"}
    )

    if start.empty or end.empty:
        raise ValueError(f"Keine Daten bei km=1.0 oder km={km_end} gefunden.")

    # Cluster-Spalte aus Start-Snapshot uebernehmen
    keep = ["Subject", "DF_start", "SF_start", cluster_col]
    merged = start[keep].merge(
        end[["Subject", "DF_end", "SF_end"]],
        on="Subject", how="inner",
    )

    color_map = _build_color_map(merged[cluster_col])
    if "Noise" in color_map:
        color_map["Noise"] = "#AAAAAA"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    method_str = _method_label(selection)
    method_suffix = f" \u2013 {method_str}" if method_str else ""
    ax.set_title(f"Erm\u00fcdungsverlauf im Dual-Axis Raum (km 1.0 \u2192 km {km_end:.1f}){method_suffix}")
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
        mpatches.Patch(
            color=color_map[lab],
            label="Noise (HDBSCAN)" if lab == "Noise" else str(lab),
        )
        for lab in sorted(color_map)
    ]
    ax.legend(handles=handles, title=cluster_col, loc="best", frameon=True)

    fig.tight_layout()
    _save(fig, out_dir, f"dual_axis_arrows{_file_suffix(selection)}.png")
    return fig, ax


# ── Plot 3 ────────────────────────────────────────────────────────────────────

def cluster_scatter(
    df: pd.DataFrame,
    *,
    cluster_col: str = "true_running_style",
    selection=None,
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
    if "Noise" in color_map:
        color_map["Noise"] = "#AAAAAA"

    fig, ax = plt.subplots(figsize=FIGSIZE)
    method_str = _method_label(selection)
    suffix = f" ({method_str})" if method_str else ""
    ax.set_title(f"Clustering-Ergebnis im Dual-Axis Raum{suffix}")
    _set_axes(ax)
    _add_region_labels(ax)

    for lab in sorted(color_map):
        sub = snap[snap[cluster_col] == lab]
        c   = color_map[lab]

        scatter_label = "Noise (HDBSCAN)" if lab == "Noise" else str(lab)
        ax.scatter(sub["DF"], sub["SF_norm"],
                   color=c, s=60, alpha=0.85, zorder=3, label=scatter_label)

        # Zentroid als X
        cx = sub["DF"].mean()
        cy = sub["SF_norm"].mean()
        ax.scatter(cx, cy,
                   marker="X", s=180, color=c,
                   edgecolors="black", linewidths=0.8,
                   zorder=5)

    ax.legend(title=cluster_col, loc="best", frameon=True)

    fig.tight_layout()
    _save(fig, out_dir, f"cluster_scatter{_file_suffix(selection)}.png")
    return fig, ax


# ── Plot 4 ────────────────────────────────────────────────────────────────────

def metrics_table(
    df_results: pd.DataFrame,
    *,
    selection=None,
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
    mask = df_results["method"] == "hdbscan"
    if mask.any():
        groups.append(("hdbscan", None, df_results[mask].copy()))

    # ── Zeilen aufbauen ───────────────────────────────────────────────────────
    display_rows: list[dict] = []
    _best_hdb_row = None  # fuer HDBSCAN-Fussnote in der Legende

    for method, linkage, df_grp in groups:
        best_rows = df_grp[df_grp["is_best"] == True]
        if best_rows.empty:
            continue

        # ── HDBSCAN: eine Zeile pro min_cluster_size (bestes rank_mean) ───────
        if method == "hdbscan":
            valid = df_grp[df_grp["silhouette"].notna()].copy()
            if valid.empty:
                continue

            # Globale beste HDBSCAN-Zeile fuer is_best-Markierung und Legende
            if "rank_mean" in best_rows.columns and best_rows["rank_mean"].notna().any():
                best_idx      = best_rows["rank_mean"].idxmin()
            else:
                best_idx      = best_rows["silhouette"].idxmax()
            _best_hdb_row = df_grp.loc[best_idx]

            mcs_values = sorted(valid["min_cluster_size"].unique())[:3]
            for mcs in mcs_values:
                mcs_rows = valid[valid["min_cluster_size"] == mcs]
                if mcs_rows.empty:
                    continue
                if "rank_mean" in mcs_rows.columns and mcs_rows["rank_mean"].notna().any():
                    r = mcs_rows.loc[mcs_rows["rank_mean"].idxmin()]
                else:
                    r = mcs_rows.loc[mcs_rows["silhouette"].idxmax()]

                row_data = [
                    "HDBSCAN", f"mcs={int(mcs)}", f"{int(r['n_clusters_found'])}*",
                    f"{r['silhouette']:.4f}",
                    f"{r['davies_bouldin']:.4f}",
                    f"{r['calinski_harabasz']:.2f}",
                ]
                display_rows.append({
                    "data":          row_data,
                    "is_best":       (r.name == best_idx),
                    "exists":        True,
                    "last_in_group": False,
                })
            if display_rows:
                display_rows[-1]["last_in_group"] = True
            continue

        # ── k-Means / Hierarchisch: best_k ± 1 ───────────────────────────────
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
    legend_h_in  = 2.15
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
        "\u2193\u202fniedriger ist besser.\n\n"

        "k\u202f=\u202fAnzahl der Cluster.\u2003"
        "mcs\u202f=\u202fmin_cluster_size (HDBSCAN: Mindestgr\u00f6\u00dfe eines Clusters)."
    )

    if _best_hdb_row is not None:
        _hdb_mcs = int(_best_hdb_row["min_cluster_size"])
        _hdb_ms  = "auto" if pd.isna(_best_hdb_row["min_samples"]) else str(int(_best_hdb_row["min_samples"]))
        legend_text += (
            f"\n\n* k automatisch durch HDBSCAN bestimmt"
            f" (min_cluster_size={_hdb_mcs}, min_samples={_hdb_ms})."
        )

    ax_leg.text(
        0.0, 1.0, legend_text,
        transform=ax_leg.transAxes,
        va="top", ha="left",
        fontsize=8, linespacing=1.55,
        color="#222222",
    )

    speed_str = "_speedber" if (selection or {}).get("speed_corrected") else "_roh"
    _save(fig, out_dir, f"metrics_table{speed_str}.png")
    plt.close(fig)
    return fig


# ── Plot 5 ────────────────────────────────────────────────────────────────────

def elbow_plot(
    df_results: pd.DataFrame,
    df_features_z: pd.DataFrame,
    cfg,
    selection=None,
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
    chosen = _method_label(selection) if selection else ""
    chosen_str = f" | Gew\u00e4hltes Verfahren: {chosen}" if chosen else ""
    ax.set_title(f"Elbow-Methode: Innerhalb-Cluster-Streuung (k-Means Inertia){chosen_str}")
    ax.set_xticks(df_km["k"].tolist())
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.legend(frameon=True)
    ax.grid(True, linewidth=0.4, alpha=0.6)
    fig.tight_layout()

    cfg.OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_PLOTS_DIR / f"elbow_plot{_file_suffix(selection)}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Elbow-Plot gespeichert -> {out}")


# ── Plot 6 ────────────────────────────────────────────────────────────────────

def dendrogram_plot(
    df_features_z: pd.DataFrame,
    selection: dict,
    cfg,
) -> None:
    """
    Plot 6: Dendrogramm des hierarchischen Clusterings.

    Nur sinnvoll bei hierarchischem Clustering. Zeigt die Merge-Hierarchie
    mit einer horizontalen gestrichelten Linie beim Schnitt fuer k Cluster.

    Parameters
    ----------
    df_features_z : Z-transformierte Feature-Tabelle mit Spalte 'Subject'
    selection     : dict mit 'linkage' und 'k'
    cfg           : config-Modul (benoetigt OUTPUT_PLOTS_DIR, RANDOM_STATE)
    """
    from scipy.cluster.hierarchy import linkage, dendrogram

    linkage_method = selection["linkage"]
    k              = selection["k"]

    X        = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)
    subjects = df_features_z["Subject"].tolist()
    n        = len(subjects)

    Z = linkage(X, method=linkage_method)

    # Schnittlinie: Mitte zwischen dem Merge der k→k-1 Reduktion
    # und dem letzten Merge innerhalb der k Cluster.
    # Z ist (n-1, 4); Zeile n-k ist der Merge von k+1 auf k Cluster,
    # Zeile n-k-1 ist der letzte Merge innerhalb der k Cluster.
    cut_upper = float(Z[n - k,     2])
    cut_lower = float(Z[n - k - 1, 2]) if k < n - 1 else 0.0
    cut_y     = (cut_upper + cut_lower) / 2.0

    # Graustufen fuer k Cluster: gleichmaessig von #000000 bis #AAAAAA
    if k > 1:
        step = 0xAA / (k - 1)
        _grey_colors = [
            f"#{int(round(i * step)):02X}{int(round(i * step)):02X}{int(round(i * step)):02X}"
            for i in range(k)
        ]
    else:
        _grey_colors = ["#000000"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        f"Dendrogramm \u2013 Hierarchisches Clustering"
        f" (Linkage: {linkage_method}, k={k})"
    )

    # Dendrogramm mit Standard-Einfärbung zeichnen (gibt k unterschiedliche Farben)
    dendrogram(
        Z,
        labels=subjects,
        ax=ax,
        color_threshold=cut_y,
        above_threshold_color="#DDDDDD",
    )

    # scipy-Farben unterhalb der Schnittlinie inventarisieren und
    # durch Graustufen ersetzen; Äste oberhalb bleiben #DDDDDD
    scipy_colors = []
    for line in ax.get_lines():
        c = line.get_color()
        if c != "#DDDDDD" and c not in scipy_colors:
            scipy_colors.append(c)

    color_remap = {sc: _grey_colors[i % len(_grey_colors)]
                   for i, sc in enumerate(scipy_colors)}
    color_remap["#DDDDDD"] = "#DDDDDD"

    for line in ax.get_lines():
        c = line.get_color()
        line.set_color(color_remap.get(c, c))

    ax.axhline(cut_y, linestyle="--", color="black", linewidth=1.2,
               label=f"Schnitt bei k={k}  (d = {cut_y:.2f})")
    ax.set_ylabel("Distanz")
    ax.tick_params(axis="x", labelsize=9, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.legend(frameon=True, fontsize=9)
    ax.grid(False)

    fig.tight_layout()

    cfg.OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_PLOTS_DIR / f"dendrogram{_file_suffix(selection)}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Dendrogramm gespeichert -> {out}")


# ── Plot 3b ───────────────────────────────────────────────────────────────────

def cluster_scatter_residuals(
    df_features_z: pd.DataFrame,
    labels: "pd.Series",
    selection: dict,
    out_dir: Optional[Path] = None,
) -> None:
    """
    Scatter im Residuen-Raum (z-transformierte DF_residual x SF_residual).
    Nur sinnvoll bei Speed-Bereinigung.
    """
    plt.rcParams.update(_BASE_RCPARAMS)

    df_labels = labels.reset_index()
    df_labels.columns = ["Subject", "cluster_label"]
    merged = df_features_z.merge(df_labels, on="Subject", how="left")

    color_map = _build_color_map(merged["cluster_label"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    method_str = _method_label(selection)
    ax.set_title(f"Clustering im Residuen-Raum\n({method_str})")
    ax.set_xlabel("DF_residual (z-standardisiert)")
    ax.set_ylabel("SF_residual (z-standardisiert)")
    ax.grid(True, linewidth=0.4, alpha=0.6)

    feat_cols = [c for c in df_features_z.columns if c != "Subject"]

    for lab in sorted(color_map):
        sub = merged[merged["cluster_label"] == lab]
        c   = color_map[lab]
        label_str = "Noise (HDBSCAN)" if lab == "Noise" else str(lab)
        ax.scatter(sub[feat_cols[0]], sub[feat_cols[1]],
                   color=c, s=60, alpha=0.85, zorder=3, label=label_str)
        cx, cy = sub[feat_cols[0]].mean(), sub[feat_cols[1]].mean()
        ax.scatter(cx, cy, marker="X", s=180, color=c,
                   edgecolors="black", linewidths=0.8, zorder=5)

    ax.legend(title="Cluster", loc="best", frameon=True)
    fig.tight_layout()

    suffix = _file_suffix(selection)
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(out_dir) / f"cluster_scatter_residuen{suffix}.png",
                    dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ── Alle Plots auf einmal ─────────────────────────────────────────────────────

def create_all_plots(
    df: pd.DataFrame,
    labels: "pd.Series",
    df_results: pd.DataFrame,
    cfg,
    selection: dict,
    df_features_z: pd.DataFrame,
) -> None:
    """
    Erzeugt alle Dual-Axis Plots mit den finalen Cluster-Labels.
    Bei hierarchischem Clustering zusaetzlich das Dendrogramm.

    Parameters
    ----------
    df            : langer Datensatz (Subject x km) ohne cluster_label
    labels        : pd.Series mit Index=Subject, Values=Cluster-Label
    df_results    : Ausgabe von evaluate_clustering_methods() + add_best_flag()
    cfg           : config-Modul (benoetigt OUTPUT_PLOTS_DIR)
    selection     : dict mit 'method', optional 'linkage', 'k'
    df_features_z : Z-transformierte Feature-Tabelle (fuer Dendrogramm)
    """
    print("\n=== Schritt 5b: Plots erstellen ===")

    cfg.OUTPUT_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df_labels  = labels.reset_index()
    df_labels.columns = ["Subject", "cluster_label"]
    df_labeled = df.merge(df_labels, on="Subject", how="left")

    print("  Plot 1: dual_axis_snapshot ...")
    dual_axis_snapshot(df_labeled, selection=selection, out_dir=cfg.OUTPUT_PLOTS_DIR)

    print("  Plot 2: dual_axis_arrows ...")
    dual_axis_arrows(df_labeled, cluster_col="cluster_label", selection=selection, out_dir=cfg.OUTPUT_PLOTS_DIR)

    print("  Plot 3: cluster_scatter (Dual-Axis Rohdaten) ...")
    cluster_scatter(df_labeled, cluster_col="cluster_label", selection=selection, out_dir=cfg.OUTPUT_PLOTS_DIR)

    if selection.get("speed_corrected"):
        print("  Plot 3b: cluster_scatter (Residuen-Raum) ...")
        cluster_scatter_residuals(df_features_z, labels, selection, cfg.OUTPUT_PLOTS_DIR)

    print("  Plot 4: metrics_table ...")
    metrics_table(df_results, selection=selection, out_dir=cfg.OUTPUT_PLOTS_DIR)

    if selection["method"] == "hierarchical":
        print("  Plot 6: dendrogram_plot ...")
        dendrogram_plot(df_features_z, selection, cfg)
    else:
        print("  Dendrogramm: uebersprungen (nur bei hierarchischem Clustering)")

    plt.close("all")
    print(f"  Plots gespeichert -> {cfg.OUTPUT_PLOTS_DIR}")
