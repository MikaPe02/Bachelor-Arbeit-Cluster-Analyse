import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.viz import plot_dual_axis_snapshot

# 1) Load your long-format dual-axis dataset
df_long = pd.read_csv(project_root / "dual_axis_dataset.txt", sep="\t")

# 2) Output path
out_path1 = project_root / "Outputs" / "Plots" / "dual_axis_scatter_km1_5.png"
out_path1.parent.mkdir(parents=True, exist_ok=True)
out_path2 = project_root / "Outputs" / "Plots" / "dual_axis_scatter_km9_5.png"
out_path2.parent.mkdir(parents=True, exist_ok=True)

# 3) Plot at km=1.5 (no clusters yet)
plot_dual_axis_snapshot(
    df_long=df_long,
    cluster_labels=None,  # Optional: if you have cluster labels to color by
    km_value=1.5,
    x_col="DF",
    y_col="SF_norm",
    title="Dual-Axis Scatter (DF vs SF_norm) at km=1.5",
    label_name="group",
    out_path=out_path1,
    show_centroids=False,
)

print("Saved:", out_path1)


# 4) Plot at km=9.5 (no clusters yet)
plot_dual_axis_snapshot(
    df_long=df_long,
    cluster_labels=None,  # Optional: if you have cluster labels to color by
    km_value=9.5,
    x_col="DF",
    y_col="SF_norm",
    title="Dual-Axis Scatter (DF vs SF_norm) at km=9.5",
    label_name="group",
    out_path=out_path2,
    show_centroids=False,
)

print("Saved:", out_path2)