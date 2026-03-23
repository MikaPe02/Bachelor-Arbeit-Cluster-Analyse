import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))

from fatigue.io import find_data_files
from fatigue.io import list_mat_variable_names
from fatigue.io import save_variable_names_to_txt


data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"

files = find_data_files(data_folder)

mat_files = [f for f in files if f.suffix.lower() == ".mat"]

print("MAT files:", len(mat_files))

if not mat_files:
    raise SystemExit("No .mat files found.")

names = list_mat_variable_names(mat_files[0])

print("Variablen in PARAMETERS.R:")

for n in names:
    print(" -", n)


output_file = "variable_names.txt"

save_variable_names_to_txt(names, output_file)

print("\nVariablen gespeichert in:")
print(output_file)