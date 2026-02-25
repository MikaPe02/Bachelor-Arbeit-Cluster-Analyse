import sys
from pathlib import Path

# add src folder to Python path
project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"

sys.path.append(str(src_path))


from fatigue.io import find_data_files


folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"

files = find_data_files(folder)

print("Gefundene Dateien:", len(files))

for f in files[:10]:
    print(f)
