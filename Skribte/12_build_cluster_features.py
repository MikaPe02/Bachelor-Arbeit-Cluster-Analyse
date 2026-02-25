import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.fatigue_metrics import build_fatigue_feature_table


# 1) Input laden
df = pd.read_csv("dual_axis_dataset.txt", sep="\t")

# 2) Feature-Tabelle bauen (1 Zeile pro Subject)
df_feat = build_fatigue_feature_table(df)

# 3) Optional: Z-Transformation (nur numerische Spalten)
numeric_cols = [c for c in df_feat.columns if c != "Subject"]
df_z = df_feat.copy()
df_z[numeric_cols] = (df_feat[numeric_cols] - df_feat[numeric_cols].mean()) / df_feat[numeric_cols].std(ddof=0)

# 4) Speichern
df_feat.to_csv("cluster_features_raw.txt", sep="\t", index=False)
df_z.to_csv("cluster_features_z.txt", sep="\t", index=False)

print("Gespeichert:")
print("- cluster_features_raw.txt")
print("- cluster_features_z.txt")
print("\nPreview:")
print(df_z)
