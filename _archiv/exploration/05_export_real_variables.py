import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))

from fatigue.io import find_data_files
from fatigue.io import export_deep_variable_names


data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"

files = find_data_files(data_folder)

mat_files = [f for f in files if f.suffix.lower() == ".mat"]

mat_file = mat_files[0]

export_deep_variable_names(mat_file, "deep_variable_lists")

print("Export fertig")