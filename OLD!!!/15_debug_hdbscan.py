import pandas as pd

df = pd.read_csv("cluster_method_comparison_full.txt", sep="\t")

df_h = df[df["method"] == "hdbscan"].copy()

print("HDBSCAN rows:", len(df_h))

# nach Anzahl Cluster (absteigend) und Noise (aufsteigend)
df_h = df_h.sort_values(["n_clusters_found", "n_noise"], ascending=[False, True])

cols = ["min_cluster_size", "min_samples", "n_clusters_found", "n_noise",
        "silhouette", "davies_bouldin", "calinski_harabasz"]

print(df_h[cols].head(20))