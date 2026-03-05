from __future__ import annotations

from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_labels_table(path: Union[str, Path], *, subject_col: str = "Subject") -> pd.DataFrame:
    """
    Load a labels table with at least: Subject, label.
    Supports .csv (comma) and .txt/.tsv (tab).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Labels file not found: {path}")

    if path.suffix.lower() in [".txt", ".tsv"]:
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)

    if subject_col not in df.columns or "label" not in df.columns:
        raise ValueError("Labels file must contain columns: Subject and label")

    return df



def _sorted_unique_labels(labels: pd.Series) -> list:
    """Sort cluster labels stably: noise (-1) first, then numeric ascending, then strings."""
    uniq = list(pd.unique(labels))

    def _key(x):
        if x == -1:
            return (-1, 0, "")
        if isinstance(x, (int, np.integer)):
            return (0, int(x), "")
        return (1, 0, str(x))

    return sorted(uniq, key=_key)


def plot_scatter_with_clusters(
    scores: pd.DataFrame,
    cluster_labels: Union[pd.Series, dict],
    *,
    subject_col: str = "Subject",
    x_col: str = "PC1",
    y_col: str = "PC2",
    title: str = "Scatter with Cluster Labels",
    label_name: str = "cluster",
    show_centroids: bool = True,
    annotate_subjects: bool = False,
    draw_zero_lines: bool = True,   # <-- NEU
    figsize: Tuple[float, float] = (7.5, 6.0),
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    2D scatter plot colored by cluster labels.

    scores: DataFrame with subject_col, x_col, y_col
    cluster_labels: mapping Subject -> label (Series with index or dict)
    """

    required = {subject_col, x_col, y_col}
    missing = required - set(scores.columns)
    if missing:
        raise ValueError(f"scores is missing required columns: {missing}")

    if isinstance(cluster_labels, dict):
        labels = pd.Series(cluster_labels, name=label_name)
        labels.index.name = subject_col
    elif isinstance(cluster_labels, pd.Series):
        labels = cluster_labels.copy()
        labels.name = label_name
        if labels.index.name is None:
            labels.index.name = subject_col
    else:
        raise TypeError("cluster_labels must be a pd.Series or a dict")

    merged = scores[[subject_col, x_col, y_col]].copy()
    merged = merged.merge(
        labels.reset_index(),
        on=subject_col,
        how="inner",
        validate="one_to_one",
    )

    if merged.empty:
        raise ValueError(
            "After merging scores and cluster labels, no rows remain. "
            "Check that Subject IDs match in both inputs."
        )

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if draw_zero_lines:
        ax.axhline(0, linewidth=0.8)
        ax.axvline(0, linewidth=0.8)

    unique_labels_sorted = _sorted_unique_labels(merged[label_name])

    for lab in unique_labels_sorted:
        sub = merged[merged[label_name] == lab]
        ax.scatter(sub[x_col], sub[y_col], label=str(lab), alpha=0.85)

        if annotate_subjects:
            for _, r in sub.iterrows():
                ax.text(r[x_col], r[y_col], str(r[subject_col]), fontsize=8)

        if show_centroids and lab != -1 and len(sub) >= 2:
            cx = float(sub[x_col].mean())
            cy = float(sub[y_col].mean())
            ax.scatter(cx, cy, marker="X", s=120)

    ax.legend(title=label_name, loc="best", frameon=True)
    ax.grid(True, linewidth=0.5)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, merged


def plot_pca_with_clusters(
    pca_scores: pd.DataFrame,
    cluster_labels: Union[pd.Series, dict],
    *,
    subject_col: str = "Subject",
    pcx: str = "PC1",
    pcy: str = "PC2",
    label_name: str = "cluster",
    title: Optional[str] = None,
    **kwargs,
):
    """Convenience wrapper for PCA plots with flexible component selection."""
    if title is None:
        title = f"PCA ({pcx} vs {pcy}) with Cluster Labels"

    return plot_scatter_with_clusters(
        scores=pca_scores,
        cluster_labels=cluster_labels,
        subject_col=subject_col,
        x_col=pcx,
        y_col=pcy,
        title=title,
        label_name=label_name,
        **kwargs,
    )

def make_km_snapshot(
    df_long: pd.DataFrame,
    *,
    km_value: float,
    subject_col: str = "Subject",
    km_col: str = "km",
    keep_cols: tuple = ("DF", "SF_norm"),
    tol: float = 1e-6,
) -> pd.DataFrame:
    """
    From a long-format dual-axis dataset (Subject x km),
    create a per-subject snapshot at a specific km value.

    Returns DataFrame with:
    Subject + keep_cols
    """

    required = {subject_col, km_col, *keep_cols}
    missing = required - set(df_long.columns)
    if missing:
        raise ValueError(f"df_long missing columns: {missing}")

    mask = (df_long[km_col] - km_value).abs() <= tol
    snap = df_long.loc[mask, [subject_col, *keep_cols]].copy()

    if snap.empty:
        raise ValueError(f"No rows found for km={km_value}")

    if snap.duplicated(subset=[subject_col]).any():
        raise ValueError(
            "Duplicate subjects found at this km. "
            "Check your dataset or tolerance."
        )

    return snap

def plot_dual_axis_snapshot(
    df_long: pd.DataFrame,
    cluster_labels: Union[pd.Series, dict, None],
    *,
    km_value: float,
    x_col: str = "DF",
    y_col: str = "SF_norm",
    subject_col: str = "Subject",
    title: Optional[str] = None,
    label_name: str = "cluster",
    out_path: Optional[Union[str, Path]] = None,
    show_centroids: bool = True,
):
    """
    Dual-Axis scatter at chosen km.
    Uses existing generic scatter function internally.
    """

    snap = make_km_snapshot(
        df_long,
        km_value=km_value,
        subject_col=subject_col,
        keep_cols=(x_col, y_col),
    )

    if title is None:
        title = f"Dual-Axis ({x_col} vs {y_col}) at km={km_value}"

    if cluster_labels is None:
        cluster_labels = pd.Series(
            ["all"] * len(snap),
            index=snap[subject_col].values
        )

    return plot_scatter_with_clusters(
        scores=snap,
        cluster_labels=cluster_labels,
        subject_col=subject_col,
        x_col=x_col,
        y_col=y_col,
        title=title,
        label_name=label_name,
        out_path=out_path,
        show_centroids=show_centroids,
        draw_zero_lines=False,
    )

def plot_dual_axis_arrows(
    df_long: pd.DataFrame,
    *,
    km_start: float,
    km_end: float,
    cluster_labels: Union[pd.Series, dict, None] = None,
    subject_col: str = "Subject",
    km_col: str = "km",
    x_col: str = "DF",
    y_col: str = "SF_norm",
    title: Optional[str] = None,
    label_name: str = "cluster",
    out_path: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    tol: float = 1e-6,
    arrow: bool = True,
    alpha: float = 0.9,
    linewidth: float = 2.0,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """
    Plot start->end change in Dual-Axis space (DF vs SF_norm) per subject.
    Each subject is a line (optionally with arrow head).

    Cluster coloring:
    - If cluster_labels is provided: one color per cluster label.
    - If cluster_labels is None: all lines are gray.
    - If label == -1 (HDBSCAN noise): colored gray by convention.

    Returns merged table with start/end coordinates per subject.
    """

    # 1) Create snapshots
    start = make_km_snapshot(
        df_long,
        km_value=km_start,
        subject_col=subject_col,
        km_col=km_col,
        keep_cols=(x_col, y_col),
        tol=tol,
    ).rename(columns={x_col: f"{x_col}_start", y_col: f"{y_col}_start"})

    end = make_km_snapshot(
        df_long,
        km_value=km_end,
        subject_col=subject_col,
        km_col=km_col,
        keep_cols=(x_col, y_col),
        tol=tol,
    ).rename(columns={x_col: f"{x_col}_end", y_col: f"{y_col}_end"})

    merged = start.merge(end, on=subject_col, how="inner", validate="one_to_one")
    if merged.empty:
        raise ValueError("No subjects overlap between start and end snapshots.")

    # 2) Prepare labels (Subject -> label)
    if cluster_labels is None:
        labels = pd.Series(["all"] * len(merged), index=merged[subject_col].values, name=label_name)
    elif isinstance(cluster_labels, dict):
        labels = pd.Series(cluster_labels, name=label_name)
    elif isinstance(cluster_labels, pd.Series):
        labels = cluster_labels.copy()
        labels.name = label_name
    else:
        raise TypeError("cluster_labels must be None, pd.Series, or dict")

    labels.index.name = subject_col
    merged = merged.merge(labels.reset_index(), on=subject_col, how="left")

    if merged[label_name].isna().any():
        merged[label_name] = merged[label_name].fillna("unlabeled")

    # 3) Build a stable color map: one color per cluster label
    unique_labels = _sorted_unique_labels(merged[label_name])

    color_map = {}
    if cluster_labels is None:
        # all gray
        for lab in unique_labels:
            color_map[lab] = "gray"
    else:
        # Use matplotlib default cycle, but assign PER LABEL (not per line)
        cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if not cycle:
            cycle = ["C0", "C1", "C2", "C3", "C4", "C5"]

        idx = 0
        for lab in unique_labels:
            if lab == -1:  # noise
                color_map[lab] = "gray"
            else:
                color_map[lab] = cycle[idx % len(cycle)]
                idx += 1

    # 4) Plot
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    if title is None:
        title = f"Dual-Axis change {km_start} → {km_end} km"
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.grid(True, linewidth=0.5)

    # Draw lines grouped by label, but with fixed color per label
    for lab in unique_labels:
        sub = merged[merged[label_name] == lab]
        c = color_map[lab]

        for _, r in sub.iterrows():
            x0, y0 = r[f"{x_col}_start"], r[f"{y_col}_start"]
            x1, y1 = r[f"{x_col}_end"], r[f"{y_col}_end"]

            ax.plot([x0, x1], [y0, y1], color=c, alpha=alpha, linewidth=linewidth)

            if arrow:
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", linewidth=linewidth, color=c, alpha=alpha),
                )

        # One legend handle per cluster
        ax.scatter([], [], color=c, label=str(lab))

    ax.legend(title=label_name, loc="best", frameon=True)

    # 5) Save optional
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")

    return fig, ax, merged
