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

FIGSIZE = (16 / 2.54, 13 / 2.54)   # 16 cm Breite, quadratisch-ähnlich
DPI     = 300
XLIM    = (0.43, 0.92)   # DF-Achse, einheitlich fuer alle drei Plots
YLIM    = (0.62, 1.15)   # SF_norm-Achse, einheitlich fuer alle drei Plots


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
    ax.set_xlabel("Duty Factor [–]")
    ax.set_ylabel("Normierte Schrittfrequenz [–]")
    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.grid(True, linewidth=0.5, alpha=0.5)



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
    _set_axes(ax)

    ax.scatter(
        snap["DF"], snap["SF_norm"],
        color="#4477AA", s=60, zorder=3, alpha=0.85,
    )

    # Subject-Labels leicht versetzt
    for _, row in snap.iterrows():
        ax.annotate(
            row["Subject"],
            xy=(row["DF"], row["SF_norm"]),
            xytext=(4, 4), textcoords="offset points",
            fontsize=6, color="#555555",
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
    _set_axes(ax)

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

    # Legende: Kreise für Cluster-Farben
    handles = [
        mlines.Line2D([0], [0], marker="o", color=color_map[lab], linestyle="None",
                      markersize=7, label="Noise (HDBSCAN)" if lab == "Noise" else str(lab))
        for lab in sorted(color_map)
    ]
    # Legende: Startpunkt + Pfeil-Erklärung
    legend_extra = [
        mlines.Line2D([0], [0], marker="o", color="#555555", linestyle="None",
                      markersize=6, label="km 1 (Start)"),
        mlines.Line2D([0], [0], marker=">", color="#555555", linestyle="-",
                      markersize=6, label="km 10 (Ende)"),
    ]
    ax.legend(handles=handles + legend_extra, title="Cluster", loc="upper right", frameon=True)

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
    _set_axes(ax)

    for lab in sorted(color_map):
        sub = snap[snap[cluster_col] == lab]
        c   = color_map[lab]

        scatter_label = "Noise (HDBSCAN)" if lab == "Noise" else str(lab)
        ax.scatter(sub["DF"], sub["SF_norm"],
                   color=c, s=55, alpha=0.85, zorder=3, label=scatter_label)

        # Zentroid als X
        cx = sub["DF"].mean()
        cy = sub["SF_norm"].mean()
        ax.scatter(cx, cy,
                   marker="X", s=160, color=c,
                   edgecolors="black", linewidths=0.8,
                   zorder=5)

    # Zentroid-Eintrag in Legende
    centroid_handle = mlines.Line2D(
        [], [], marker="X", color="gray", linestyle="None",
        markersize=8, markeredgecolor="black", markeredgewidth=0.8,
        label="Zentroid",
    )
    handles, labels_leg = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [centroid_handle], labels=labels_leg + ["Zentroid"],
              title="Cluster", loc="best", frameon=True)

    fig.tight_layout()
    _save(fig, out_dir, f"cluster_scatter{_file_suffix(selection)}.png")
    return fig, ax


# ── Plot 4 ────────────────────────────────────────────────────────────────────────────────

def metrics_table(
    df_results: pd.DataFrame,
    *,
    selection=None,
    out_dir=None,
) -> plt.Figure:
    # APA-Stil Tabelle der Clustering-Validierungsmetriken. Kein Titel, keine Legende.
    plt.rcParams.update(_BASE_RCPARAMS)

    sil_col = "silhouette"
    db_col  = "davies_bouldin"
    ch_col  = "calinski_harabasz"
    nf_col  = "n_clusters_found"

    COL_LABELS = [
        "Methode", "Linkage", "k",
        "Silhouette ↑", "D.-Bouldin ↓", "Cal.-Harabasz ↑",
    ]
    COL_X     = [0.01, 0.22, 0.33, 0.48, 0.65, 0.83]
    COL_ALIGN = ["left", "center", "center", "center", "center", "center"]

    # Gruppen aufbauen — wenn Ward gewählt, nur Ward anzeigen
    sel_method  = (selection or {}).get("method", "")
    sel_linkage = (selection or {}).get("linkage", "")
    ward_only   = sel_method == "hierarchical" and sel_linkage == "ward"

    groups = []
    if ward_only:
        mask = (df_results["method"] == "hierarchical") & (df_results["linkage"] == "ward")
        if mask.any():
            groups.append(("hierarchical", "ward", df_results[mask].copy()))
    else:
        for lnk in ["ward", "complete", "average", "single"]:
            mask = (df_results["method"] == "hierarchical") & (df_results["linkage"] == lnk)
            if mask.any():
                groups.append(("hierarchical", lnk, df_results[mask].copy()))
        for meth in ["kmeans", "hdbscan"]:
            mask = df_results["method"] == meth
            if mask.any():
                groups.append((meth, None, df_results[mask].copy()))

    display_rows: list[dict] = []

    for method, linkage, df_grp in groups:
        best_rows = df_grp[df_grp["is_best"] == True]
        if best_rows.empty:
            continue

        if method == "hdbscan":
            valid = df_grp[df_grp[sil_col].notna()].copy()
            if valid.empty:
                continue
            if "rank_mean" in best_rows.columns and best_rows["rank_mean"].notna().any():
                best_idx = best_rows["rank_mean"].idxmin()
            else:
                best_idx = best_rows[sil_col].idxmax()
            for mcs in sorted(valid["min_cluster_size"].unique())[:3]:
                mcs_rows = valid[valid["min_cluster_size"] == mcs]
                if mcs_rows.empty:
                    continue
                if "rank_mean" in mcs_rows.columns and mcs_rows["rank_mean"].notna().any():
                    r = mcs_rows.loc[mcs_rows["rank_mean"].idxmin()]
                else:
                    r = mcs_rows.loc[mcs_rows[sil_col].idxmax()]
                display_rows.append({
                    "data": [
                        "HDBSCAN", f"mcs = {int(mcs)}", str(int(r[nf_col])),
                        f"{r[sil_col]:.3f}", f"{r[db_col]:.3f}", f"{r[ch_col]:.1f}",
                    ],
                    "is_best": r.name == best_idx,
                    "is_selected": False,
                    "exists": True,
                    "last_in_group": False,
                })
            if display_rows:
                display_rows[-1]["last_in_group"] = True
            continue

        best_k   = int(best_rows.iloc[0]["k"])
        meth_lbl = "Hierarchisch" if method == "hierarchical" else "k-Means"
        lnk_lbl  = linkage.capitalize() if linkage else "—"

        # Sicherstellen dass die gewählte k auch in der Tabelle erscheint
        sel_k = int(selection["k"]) if (
            selection and selection.get("method") == method
            and (linkage is None or selection.get("linkage") == linkage)
        ) else None
        if ward_only and sel_k is not None:
            # Nur k-1, k, k+1 relativ zur gewählten k zeigen
            k_vals = sorted(set([sel_k - 1, sel_k, sel_k + 1]))
        else:
            k_vals = sorted(set([best_k - 1, best_k, best_k + 1] + ([sel_k] if sel_k else [])))

        for k_val in k_vals:
            match  = df_grp[df_grp["k"] == k_val]
            exists = not match.empty
            if exists:
                r    = match.iloc[0]
                data = [
                    meth_lbl, lnk_lbl, str(k_val),
                    f"{r[sil_col]:.3f}", f"{r[db_col]:.3f}", f"{r[ch_col]:.1f}",
                ]
            else:
                data = [meth_lbl, lnk_lbl, str(k_val), "—", "—", "—"]
            is_selected = (sel_k is not None and k_val == sel_k)
            display_rows.append({
                "data": data,
                "is_best": k_val == best_k,
                "is_selected": is_selected,
                "exists": exists,
                "last_in_group": False,
            })
        display_rows[-1]["last_in_group"] = True

    # Figur
    n_data   = len(display_rows)
    row_h_in = 0.30
    pad_in   = 0.20
    fig_w    = 20 / 2.54   # etwas breiter für Spaltenabstand
    fig_h    = pad_in + (n_data + 1) * row_h_in + pad_in
    fig      = plt.figure(figsize=(fig_w, fig_h))

    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    top      = fig_h - pad_in
    header_y = top - row_h_in / 2
    data_ys  = [top - row_h_in * (i + 1.5) for i in range(n_data)]
    data_bot = [top - row_h_in * (i + 2.0) for i in range(n_data)]

    def _hline(y, lw, color="black"):
        ax.plot([0, 1], [y, y], color=color, linewidth=lw,
                clip_on=False, solid_capstyle="butt", zorder=4)

    # Toprule
    _hline(top, 1.5)
    # Header
    for j, lbl in enumerate(COL_LABELS):
        ax.text(COL_X[j], header_y, lbl,
                ha=COL_ALIGN[j], va="center",
                fontsize=10, fontweight="bold", color="black", zorder=3)
    # Midrule
    _hline(top - row_h_in, 1.0)

    for i, meta in enumerate(display_rows):
        y      = data_ys[i]
        ybot   = data_bot[i]
        ytop   = ybot + row_h_in
        color  = "black"  if meta["exists"]     else "#AAAAAA"
        weight = "bold"   if meta["is_best"] or meta.get("is_selected") else "normal"

        # Gewählte Konfiguration: hellgrauer Hintergrund + Markierung
        if meta.get("is_selected"):
            ax.add_patch(mpatches.Rectangle(
                (0, ybot), 1.0, row_h_in,
                facecolor="#E8F0FB", edgecolor="none", zorder=1,
            ))

        for j, val in enumerate(meta["data"]):
            ax.text(COL_X[j], y, val,
                    ha=COL_ALIGN[j], va="center",
                    fontsize=10, color=color, fontweight=weight, zorder=3)

        # Markierungssymbol rechts außen für gewählte Zeile
        if meta.get("is_selected"):
            ax.text(0.98, y, "◄ gewählt",
                    ha="right", va="center",
                    fontsize=8, color="#2255AA", fontstyle="italic", zorder=3)

        if i == n_data - 1:
            _hline(ybot, 1.5)
        elif meta["last_in_group"]:
            _hline(ybot, 0.8, "#555555")
        else:
            _hline(ybot, 0.4, "#BBBBBB")

    speed_str = "_speedber" if (selection or {}).get("speed_corrected") else "_roh"
    if out_dir is not None:
        from pathlib import Path as _Path
        _out = _Path(out_dir)
        _out.mkdir(parents=True, exist_ok=True)
        fig.savefig(_out / f"metrics_table{speed_str}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return fig

# ── Plot 5 ────────────────────────────────────────────────────────────────────

def elbow_plot(
    df_results: pd.DataFrame,
    df_features_z: pd.DataFrame,
    cfg,
    selection=None,
    out_dir=None,
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

    fig, ax = plt.subplots(figsize=(16 / 2.54, 8 / 2.54))
    ax.plot(df_km["k"], df_km["inertia_pct"], marker="o", linewidth=2.0,
            color="#4477AA", markersize=7)
    ax.set_xlabel("Anzahl Cluster (k)")
    ax.set_ylabel("WCSS (normiert, %)")
    ax.set_xticks(df_km["k"].tolist())
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(decimals=0))
    ax.grid(True, linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    fig.subplots_adjust(left=0.12)

    plot_dir = Path(out_dir) if out_dir is not None else cfg.OUTPUT_PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"elbow_plot{_file_suffix(selection)}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Elbow-Plot gespeichert -> {out}")


# ── Plot 6 ────────────────────────────────────────────────────────────────────

def dendrogram_plot(
    df_features_z: pd.DataFrame,
    selection: dict,
    cfg,
    labels: "pd.Series | None" = None,
    out_dir=None,
) -> None:
    """
    Plot 6: Dendrogramm des hierarchischen Clusterings.

    Nur sinnvoll bei hierarchischem Clustering. Zeigt die Merge-Hierarchie
    mit einer horizontalen gestrichelten Linie beim Schnitt fuer k Cluster.
    Cluster-Aeste werden mit denselben Farben wie die anderen Plots eingefaerbt
    (Paul-Tol-Palette), sofern labels uebergeben werden.

    Parameters
    ----------
    df_features_z : Z-transformierte Feature-Tabelle mit Spalte 'Subject'
    selection     : dict mit 'linkage' und 'k'
    cfg           : config-Modul (benoetigt OUTPUT_PLOTS_DIR, RANDOM_STATE)
    labels        : pd.Series mit Index=Subject, Values=Cluster-Label (fuer konsistente Farben)
    """
    from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

    linkage_method = selection["linkage"]
    k              = selection["k"]

    X        = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)
    subjects = df_features_z["Subject"].tolist()
    n        = len(subjects)

    Z = linkage(X, method=linkage_method)

    # Schnittlinie: Mitte zwischen dem Merge der k→k-1 Reduktion
    # und dem letzten Merge innerhalb der k Cluster.
    cut_upper = float(Z[n - k,     2])
    cut_lower = float(Z[n - k - 1, 2]) if k < n - 1 else 0.0
    cut_y     = (cut_upper + cut_lower) / 2.0

    # Y-Achsen-Beschriftung je nach Linkage-Methode
    _ylabel_map = {
        "ward":     "Fusionierungsdistanz (Ward)",
        "complete": "Fusionierungsdistanz (Complete Linkage)",
        "average":  "Fusionierungsdistanz (Average Linkage)",
        "single":   "Fusionierungsdistanz (Single Linkage)",
    }
    ylabel = _ylabel_map.get(linkage_method, "Fusionierungsdistanz")

    # Cluster-Zuordnung: Blatt-Index → echtes Cluster-Label → Palette-Farbe
    cluster_ids = fcluster(Z, k, criterion="maxclust")  # 1-basiert, Länge n
    # Palette nach sortierten echten Labels aufbauen (identisch zu _build_color_map)
    if labels is not None:
        label_series = labels.reset_index()
        label_series.columns = ["Subject", "cluster_label"]
        sorted_real = sorted(set(str(v) for v in labels.values), key=str)
        real_to_color = {lab: _PALETTE[i % len(_PALETTE)] for i, lab in enumerate(sorted_real)}
        # Blatt-Index → echtes Label
        leaf_to_color: dict[int, str] = {}
        for i, s in enumerate(subjects):
            row = label_series.loc[label_series["Subject"] == s, "cluster_label"]
            if not row.empty:
                leaf_to_color[i] = real_to_color.get(str(row.iloc[0]), "#CCCCCC")
            else:
                leaf_to_color[i] = "#CCCCCC"
        # scipy-Cluster-ID (1-basiert) → Farbe über repräsentatives Blatt
        scipy_id_to_color: dict[int, str] = {}
        for i, cid in enumerate(cluster_ids):
            if cid not in scipy_id_to_color:
                scipy_id_to_color[cid] = leaf_to_color.get(i, "#CCCCCC")
    else:
        scipy_id_to_color = {i + 1: _PALETTE[i % len(_PALETTE)] for i in range(k)}

    # link_color_func: scipy übergibt den Index des Merge-Knotens
    # Knoten deren Merge-Distanz >= cut_y ist → grau (oberhalb des Schnitts)
    def _link_color(node_id: int) -> str:
        if node_id < n:
            # Blattknoten: immer Clusterfarbe
            cid = int(cluster_ids[node_id])
            return scipy_id_to_color.get(cid, "#CCCCCC")
        merge_row = node_id - n
        if merge_row >= len(Z):
            return "#CCCCCC"
        merge_dist = float(Z[merge_row, 2])
        if merge_dist >= cut_y:
            return "#CCCCCC"
        # Unterhalb des Schnitts: Farbe anhand des ersten Blatts links
        left = int(Z[merge_row, 0])
        leaf = left
        while leaf >= n:
            leaf = int(Z[leaf - n, 0])
        cid = int(cluster_ids[leaf])
        return scipy_id_to_color.get(cid, "#CCCCCC")

    fig, ax = plt.subplots(figsize=(24 / 2.54, 12 / 2.54))

    dendrogram(
        Z,
        labels=subjects,
        ax=ax,
        color_threshold=cut_y,
        above_threshold_color="#CCCCCC",
        link_color_func=_link_color,
    )

    ax.axhline(cut_y, linestyle="--", color="black", linewidth=1.5)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Proband")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")
    ax.grid(False)

    # Legende: Cluster-Farben + Schnittlinie
    legend_handles = []
    if labels is not None:
        for i, lab in enumerate(sorted_real):
            color = real_to_color[lab]
            legend_handles.append(
                mpatches.Patch(color=color, label=f"Cluster {lab}")
            )
    else:
        for i in range(k):
            legend_handles.append(
                mpatches.Patch(color=_PALETTE[i % len(_PALETTE)], label=f"Cluster {i+1}")
            )
    legend_handles.append(
        plt.Line2D([0], [0], color="black", linewidth=1.5, linestyle="--",
                   label=f"Schnitt k = {k}")
    )
    ax.legend(handles=legend_handles, frameon=True, fontsize=10, loc="upper right")

    fig.tight_layout()

    plot_dir = Path(out_dir) if out_dir is not None else cfg.OUTPUT_PLOTS_DIR
    plot_dir.mkdir(parents=True, exist_ok=True)
    out = plot_dir / f"dendrogram{_file_suffix(selection)}.png"
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
    ax.set_xlabel("Duty Factor Residual [z]")
    ax.set_ylabel("Norm. Schrittfrequenz Residual [z]")
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


def fatigue_cluster_scatter(
    df_f_z: "pd.DataFrame",
    labels: "pd.Series",
    selection: dict,
    out_dir: "Path",
) -> None:
    """
    Scatter-Plot der Fatigue-Cluster im z-transformierten Feature-Raum.
    X-Achse: Delta_DF_z oder Slope_DF_z, Y-Achse: Delta_SF_z oder Slope_SF_z.
    """
    import pandas as pd

    suffix   = _file_suffix(selection)
    method   = _method_label(selection)

    df_plot = df_f_z.set_index("Subject").copy()
    df_plot["cluster"] = labels.astype(str)

    # Achsen: erste zwei Feature-Spalten (nach Subject)
    feature_cols = [c for c in df_f_z.columns if c != "Subject"]
    if len(feature_cols) < 2:
        return
    xcol, ycol = feature_cols[0], feature_cols[1]

    cluster_ids = sorted(df_plot["cluster"].unique(), key=str)
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cluster_ids))]

    fig, ax = plt.subplots(figsize=(7, 6))

    for cid, color in zip(cluster_ids, colors):
        sub = df_plot[df_plot["cluster"] == cid]
        ax.scatter(sub[xcol], sub[ycol], color=color, label=f"Cluster {cid}",
                   s=60, alpha=0.85, edgecolors="white", linewidths=0.5)

    ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_xlabel(f"{xcol} [z]")
    ax.set_ylabel(f"{ycol} [z]")
    ax.legend(fontsize=9, framealpha=0.8)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"fatigue_cluster_scatter{suffix}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> Fatigue-Cluster-Scatter: {out}")


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
        dendrogram_plot(df_features_z, selection, cfg, labels=labels)
    else:
        print("  Dendrogramm: uebersprungen (nur bei hierarchischem Clustering)")

    plt.close("all")
    print(f"  Plots gespeichert -> {cfg.OUTPUT_PLOTS_DIR}")
