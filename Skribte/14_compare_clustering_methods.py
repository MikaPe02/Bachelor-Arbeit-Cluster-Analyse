import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.clustering_eval import evaluate_clustering_methods, add_best_flag

df_z = pd.read_csv("cluster_features_z.txt", sep="\t")

df_eval = evaluate_clustering_methods(df_z, k_range=range(2, 7))

# Markierung hinzufügen
df_eval_marked = add_best_flag(df_eval)

# Optional: nur “schön” sortieren (nicht nach Qualität, nur nach Struktur)
df_eval_marked = df_eval_marked.sort_values(["method", "linkage", "k"], ascending=True)

df_eval_marked.to_csv("cluster_method_comparison_full.txt", sep="\t", index=False)

print("Saved: cluster_method_comparison_full.txt")
print(df_eval_marked.head(15))