import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.pca_utils import compute_pca

# --- Output paths (clean project structure) ---
out_dir = project_root / "Outputs" / "PCA"
out_dir.mkdir(parents=True, exist_ok=True)

# 1) Daten laden
df_z = pd.read_csv("cluster_features_z.txt", sep="\t")

# 2) Dual-Axis-Fatigue-Features auswählen
dual_axis_cols = ["Delta_DF", "Slope_DF", "Delta_SF", "Slope_SF"]

# 3) PCA berechnen
df_pca, var_ratio, loadings = compute_pca(
    df_z,
    n_components=2,
    feature_cols=dual_axis_cols
)

# 4) Speichern (clean outputs)
df_pca.to_csv(out_dir / "pca_scores_dual_axis.csv", index=False)
loadings.to_csv(out_dir / "pca_loadings_dual_axis.csv", index=True)

with open(out_dir / "pca_explained_variance_dual_axis.txt", "w", encoding="utf-8") as f:
    f.write("Features used:\n")
    for c in dual_axis_cols:
        f.write(f"- {c}\n")
    f.write(f"\nPC1: {var_ratio[0]:.4f}\n")
    f.write(f"PC2: {var_ratio[1]:.4f}\n")

print("Saved to:", out_dir)
print("- pca_scores_dual_axis.csv")
print("- pca_loadings_dual_axis.csv")
print("- pca_explained_variance_dual_axis.txt")
print(f"Explained variance PC1={var_ratio[0]:.3f}, PC2={var_ratio[1]:.3f}")

# 5) Plot
plt.figure()
plt.scatter(df_pca["PC1"], df_pca["PC2"])

for _, r in df_pca.iterrows():
    plt.text(r["PC1"], r["PC2"], r["Subject"], fontsize=8)

plt.xlabel(f"PC1 ({var_ratio[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({var_ratio[1]*100:.1f}% var)")
plt.title("PCA (Dual-Axis fatigue features only)")

plt.savefig(out_dir / "pca_plot_dual_axis.png", dpi=300, bbox_inches="tight")
plt.show()