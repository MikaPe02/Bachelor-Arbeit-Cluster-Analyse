from pathlib import Path
import re
import sys
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = ROOT / "Beispiel Daten" / "P61_km01_0.mat"

data = loadmat(MAT_PATH, simplify_cells=True)
R = data["PARAMETERS"]["R"]
all_feats = list(R.keys())

BASE = {"ContactTime", "FlightTimes", "StepFrequency"}

# Winkel-Regeln: nur RIGHT + ANKLE/KNEE/HIP + X/Y/Z + Endungen
pattern = re.compile(
    r"^ANGLES_RIGHT_(ANKLE|KNEE|HIP)_[XYZ]_(TD|TO|MAX|MIN|INT_NET)$"
)

wl = sorted([f for f in all_feats if (f in BASE or pattern.match(f))])

print("Total available features:", len(all_feats))
print("Whitelist size:", len(wl))
print("\nFirst 30 whitelist entries:")
for x in wl[:30]:
    print(" ", x)

# Speichern, damit freeze möglich ist
out_path = ROOT / "Skribte" / "whitelist.txt"
out_path.write_text("\n".join(wl), encoding="utf-8")
print("\nSaved whitelist to:", out_path)
