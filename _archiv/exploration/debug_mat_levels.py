import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))

from fatigue.io import load_mat
from fatigue.io import find_data_files


data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"

files = find_data_files(data_folder)

mat_files = [f for f in files if f.suffix.lower() == ".mat"]

mat_file = mat_files[0]

print("MAT FILE:")
print(mat_file)

mat = load_mat(mat_file)

print("\nTOP LEVEL KEYS:\n")

for key in mat:

    obj = mat[key]

    print("KEY:", key)
    print("TYPE:", type(obj))

    if hasattr(obj, "__dict__"):
        print("FIELDS:", list(obj.__dict__.keys())[:10])

    elif hasattr(obj, "dtype") and obj.dtype.names:
        print("DTYPE FIELDS:", obj.dtype.names[:10])

    else:
        print("NO FIELDS FOUND")

    print("\n")

    