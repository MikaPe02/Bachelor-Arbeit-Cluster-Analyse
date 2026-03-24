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
