import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))

from fatigue.io import find_data_files, parse_filename_info
from fatigue.features import dual_axis_from_mat


leg_length = 1.0

data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"

files = find_data_files(data_folder)

mat_files = [f for f in files if f.suffix.lower() == ".mat"]

rows = []

for f in mat_files:

    info = parse_filename_info(f)

    dual = dual_axis_from_mat(f, leg_length)

    rows.append(
        (
            info["subject"],
            info["km"],
            dual["DF"],
            dual["SF_norm"]
        )
    )


rows.sort()

output_file = "dual_axis_dataset.txt"

with open(output_file, "w") as txt:

    txt.write("Subject\tkm\tDF\tSF_norm\n")

    for r in rows:
        txt.write(f"{r[0]}\t{r[1]}\t{r[2]:.6f}\t{r[3]:.6f}\n")


print("Dataset gespeichert:")
print(output_file)

