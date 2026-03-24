from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = ROOT / "Beispiel Daten" / "P61_km01_0.mat"

data = loadmat(MAT_PATH, simplify_cells=True)

def info(x):
    if isinstance(x, np.ndarray):
        return f"ndarray shape={x.shape}, dtype={x.dtype}"
    return f"type={type(x).__name__}"

keys_of_interest = ["PARAMETERS", "WINKEL", "KINETICS", "CONTACT", "NORMAL", "LABELS", "FRAME", "FP", "JOINT", "MARKERS"]

print("MAT_PATH:", MAT_PATH)
print("Top-level keys:", list(data.keys()))
print("\n--- Key info ---")
for k in keys_of_interest:
    if k in data:
        print(f"{k}: {info(data[k])}")
    else:
        print(f"{k}: MISSING")
