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

