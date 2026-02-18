from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.insert(0, str(SRC))

from fatigue.io import build_fatigue_vector

MAT_PATH = ROOT / "Beispiel Daten" / "P61_km01_0.mat"

WHITELIST = [
    "ContactTime",
    "FlightTimes",
    "StepFrequency",
    "ANGLES_RIGHT_ANKLE_X_MAX",
    "ANGLES_RIGHT_ANKLE_X_MIN",
]

vec = build_fatigue_vector(MAT_PATH, WHITELIST, early_frac=0.2)

print("MAT:", MAT_PATH.name)
print("Whitelist size:", len(WHITELIST))
print("Fatigue vector length:", len(vec))
print("\nVector head:")
print(vec.head(10))

