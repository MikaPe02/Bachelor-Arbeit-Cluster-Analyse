import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))

from fatigue.io import find_data_files
from fatigue.io import export_mat_structure_to_txt


data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"

files = find_data_files(data_folder)

mat_files = [f for f in files if f.suffix.lower() == ".mat"]

if not mat_files:
    raise SystemExit("No MAT files found")

mat_file = mat_files[0]

output_file = "mat_structure.txt"

export_mat_structure_to_txt(mat_file, output_file)

print("Export fertig:")
print(output_file)