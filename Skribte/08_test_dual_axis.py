import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))

from fatigue.features import dual_axis_from_mat
from fatigue.io import find_data_files


# Test-Beinlänge
leg_length = 1.0


data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"

files = find_data_files(data_folder)

mat_files = [f for f in files if f.suffix.lower() == ".mat"]

test_file = mat_files[0]

print("Test file:")
print(test_file)
print()


result = dual_axis_from_mat(test_file, leg_length)


print("Results:")
print("Duty Factor:", result["DF"])
print("SF_norm:", result["SF_norm"])