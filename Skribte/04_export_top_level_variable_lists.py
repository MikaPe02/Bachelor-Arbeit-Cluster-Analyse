import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.io import find_data_files, export_top_level_fields_to_txt


data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"
out_dir = Path(__file__).resolve().parent / "mat_variable_lists"

files = find_data_files(data_folder)
mat_files = [f for f in files if f.suffix.lower() == ".mat"]

if not mat_files:
    raise SystemExit("No .mat files found.")

mat_file = mat_files[0]
export_top_level_fields_to_txt(mat_file, out_dir)

print("Export fertig. Output-Ordner:")
print(out_dir)

