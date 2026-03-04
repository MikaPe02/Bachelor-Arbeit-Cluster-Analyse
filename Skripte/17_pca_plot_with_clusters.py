import pandas as pd
import sys
from pathlib import Path
sys.path.append(r"C:\Users\Mika\Uni\BA\Bachelor-Arbeit-Cluster-Analyse\src")

from fatigue.viz import plot_pca_with_clusters

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))


pca_scores = pd.read_csv(r"C:\Users\Mika\Uni\BA\Outputs\PCA\pca_scores_dual_axis.csv")

labels_df = pd.read_csv(r"C:\Users\Mika\Uni\BA\Skripte\ward_labels_k3.csv")
labels = pd.Series(labels_df["cluster"].values, index=labels_df["Subject"].values)

plot_pca_with_clusters(
    pca_scores=pca_scores,
    cluster_labels=labels,
    pcx="PC2",
    pcy="PC3",
    label_name="ward_k3",
    out_path=r"C:\Users\Mika\Uni\BA\Skripte\pca_plot_ward_k3_PC2_PC3.png",
)

