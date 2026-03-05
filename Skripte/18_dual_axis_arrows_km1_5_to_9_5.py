import sys
from pathlib import Path
import pandas as pd
import numpy as np


project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.viz import plot_dual_axis_arrows

df_long = pd.read_csv(project_root / "dual_axis_dataset.txt", sep="\t")

out_path = project_root / "Outputs" / "Plots" / "dual_axis_arrows_km1_5_to_9_5.png"
out_path.parent.mkdir(parents=True, exist_ok=True)


# 1) Create reproducible dummy clusters (3 clusters)
subjects = sorted(df_long["Subject"].unique())
rng = np.random.default_rng(42)

cluster_labels = pd.Series(
    rng.integers(0, 3, size=len(subjects)),
    index=subjects,
    name="cluster"
)

# 2) Plot with clusters
plot_dual_axis_arrows(
    df_long=df_long,
    km_start=1.5,
    km_end=9.5,
    cluster_labels=cluster_labels,
    label_name="cluster",
    out_path=out_path,
)

