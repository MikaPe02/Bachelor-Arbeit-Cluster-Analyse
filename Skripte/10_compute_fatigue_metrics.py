import sys
from pathlib import Path
import pandas as pd


# Projektpfad hinzufügen
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))


from fatigue.fatigue_metrics import compute_subject_fatigue


file = "dual_axis_dataset.txt"

df = pd.read_csv(file, sep="\t")


subjects = df["Subject"].unique()

rows = []

for s in subjects:

    df_s = df[df["Subject"] == s]

    metrics = compute_subject_fatigue(df_s)

    row = {"Subject": s}

    row.update(metrics)

    rows.append(row)


df_out = pd.DataFrame(rows)

df_out.to_csv("fatigue_metrics.txt", sep="\t", index=False)

print(df_out)