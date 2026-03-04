from pathlib import Path
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = ROOT / "Beispiel Daten" / "P61_km01_0.mat"

data = loadmat(MAT_PATH, simplify_cells=True)

hdr = data["NORMAL"]["HEADER"]
print("NORMAL.HEADER type:", type(hdr).__name__)
print("NORMAL.HEADER shape:", getattr(hdr, "shape", None))
print("NORMAL.HEADER preview:", hdr[:10])

R = data["PARAMETERS"]["R"]
print("\n#features in PARAMETERS.R:", len(R))

# Beispiele: 3 Features anschauen
examples = ["ContactTime", "StepFrequency", "ANGLES_RIGHT_ANKLE_X_MAX"]
for k in examples:
    if k in R:
        v = R[k]
        print(f"\n{k}: shape={v.shape}")
        print("  first 5:", np.array(v[:5]).ravel())
        print("  last 5 :", np.array(v[-5:]).ravel())
    else:
        print(f"\n{k}: MISSING")
