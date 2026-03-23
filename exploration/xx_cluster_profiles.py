# Skribte/09_cluster_profiles.py
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Skribte" / "fatigue_dataset.csv"
RUN_DIR = ROOT / "Skribte" / "cluster_runs"

# ---- Choose ONE labels file to interpret (set this!) ----
# Example:
# LABELS_FILE = "labels_kmeans_k3_pca90.csv"
LABELS_FILE = "labels_kmeans_k3_pca90.csv"
# --------------------------------------------------------

X = pd.read_csv(DATA_PATH, index_col=0)
X = X.drop(columns=X.columns[X.isna().any()])

labels_path = RUN_DIR / LABELS_FILE
labels = pd.read_csv(labels_path, index_col=0).iloc[:, 0].rename("cluster")

df = X.join(labels)

# If HDBSCAN: cluster=-1 is noise. Keep it separate or drop it:
# df = df[df["cluster"] != -1]

# Cluster summaries (robust & interpretable)
median = df.groupby("cluster").median(numeric_only=True)
q25 = df.groupby("cluster").quantile(0.25, numeric_only=True)
q75 = df.groupby("cluster").quantile(0.75, numeric_only=True)
iqr = q75 - q25

# Cluster sizes
sizes = df["cluster"].value_counts().sort_index()

OUT_PROFILES = RUN_DIR / f"profiles_median_{LABELS_FILE.replace('.csv','')}.csv"
OUT_IQR = RUN_DIR / f"profiles_iqr_{LABELS_FILE.replace('.csv','')}.csv"
OUT_SIZES = RUN_DIR / f"cluster_sizes_{LABELS_FILE.replace('.csv','')}.csv"

median.to_csv(OUT_PROFILES)
iqr.to_csv(OUT_IQR)
sizes.to_csv(OUT_SIZES, header=["n_subjects"])

print("Saved:", OUT_PROFILES)
print("Saved:", OUT_IQR)
print("Saved:", OUT_SIZES)
print("\nCluster sizes:")
print(sizes)
