import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.clustering_eval import evaluate_clustering_methods, add_best_flag

df_z = pd.read_csv("cluster_features_z.txt", sep="\t")

df_eval = evaluate_clustering_methods(
    df_z,
    k_range=range(2, 7),
    include_hdbscan=True,
    hdbscan_min_cluster_sizes=[2, 3, 4],
    hdbscan_min_samples=[None, 1, 2, 3],
)

df_eval = add_best_flag(df_eval)

# nur “strukturiert” sortieren (nicht nach Qualität)
df_eval = df_eval.sort_values(
    ["method", "linkage", "k", "min_cluster_size", "min_samples"],
    ascending=True
)

df_eval.to_csv("cluster_method_comparison_full.txt", sep="\t", index=False)

print("Saved: cluster_method_comparison_full.txt")
print("\nBest rows (is_best=True):")
print(df_eval[df_eval["is_best"] == True][
    ["method", "linkage", "k", "min_cluster_size", "min_samples",
     "n_clusters_found", "n_noise", "silhouette", "davies_bouldin", "calinski_harabasz"]
])