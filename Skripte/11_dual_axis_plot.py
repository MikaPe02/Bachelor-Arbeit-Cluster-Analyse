import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))


file = "dual_axis_dataset.txt"

df = pd.read_csv(file, sep="\t")


subjects = df["Subject"].unique()


plt.figure()

for s in subjects:

    df_s = df[df["Subject"] == s].sort_values("km")

    SF = df_s["SF_norm"].values
    DF = df_s["DF"].values

    # Startpunkt
    plt.scatter(SF[0], DF[0], label=f"{s} start")

    # Endpunkt
    plt.scatter(SF[-1], DF[-1], marker="x", label=f"{s} end")

    # Pfeil
    plt.arrow(
        SF[0],
        DF[0],
        SF[-1]-SF[0],
        DF[-1]-DF[0],
        length_includes_head=True,
        head_width=0.005
    )

plt.xlabel("SF_norm")
plt.ylabel("Duty Factor")

plt.title("Dual Axis Running Style Evolution")

plt.legend()
plt.show()

