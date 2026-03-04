import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.pca_utils import compute_pca


# 1) Daten laden
df_z = pd.read_csv("cluster_features_z.txt", sep="\t")


# 2) PCA berechnen
df_pca, var_ratio, loadings = compute_pca(df_z, n_components=2)


# 3) Speichern
df_pca.to_csv("pca_scores.txt", sep="\t", index=False)
loadings.to_csv("pca_loadings.txt", sep="\t")

with open("pca_explained_variance.txt", "w") as f:
    f.write(f"PC1: {var_ratio[0]:.4f}\n")
    f.write(f"PC2: {var_ratio[1]:.4f}\n")

print("Saved:")
print("- pca_scores.txt")
print("- pca_loadings.txt")
print("- pca_explained_variance.txt")
print(f"Explained variance PC1={var_ratio[0]:.3f}, PC2={var_ratio[1]:.3f}")


# 4) Plot
plt.figure()

plt.scatter(df_pca["PC1"], df_pca["PC2"])

for _, r in df_pca.iterrows():
    plt.text(r["PC1"], r["PC2"], r["Subject"], fontsize=8)

plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
plt.title("PCA of fatigue feature space (z-transformed)")

plt.savefig("pca_plot.png", dpi=300)
plt.show()

