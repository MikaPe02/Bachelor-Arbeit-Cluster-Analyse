from pathlib import Path
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]      # .../BA
MAT_PATH = ROOT / "Beispiel Daten" / "P61_km01_0.mat"

data = loadmat(MAT_PATH, simplify_cells=True)

def show_dict(d, name, max_keys=30):
    if not isinstance(d, dict):
        print(f"{name}: type={type(d).__name__}")
        return
    keys = list(d.keys())
    print(f"\n{name}: dict with {len(keys)} keys")
    for k in keys[:max_keys]:
        v = d[k]
        t = type(v).__name__
        extra = ""
        # kleine Zusatzinfos ohne numpy-Import
        if hasattr(v, "shape"):
            extra = f", shape={v.shape}"
        print(f"  - {k}: {t}{extra}")
    if len(keys) > max_keys:
        print(f"  ... ({len(keys)-max_keys} more)")

# Top-level biomech blocks
blocks = ["PARAMETERS", "WINKEL", "KINETICS", "CONTACT", "NORMAL", "LABELS"]

print("MAT_PATH:", MAT_PATH)
for b in blocks:
    show_dict(data.get(b), b)

# optional: zweite Ebene für zwei wichtigste Blöcke
for b in ["PARAMETERS", "WINKEL"]:
    d = data.get(b)
    if isinstance(d, dict):
        # nimm die ersten 3 keys und zeig deren Unterkeys (falls dict)
        for k in list(d.keys())[:3]:
            show_dict(d[k], f"{b}.{k}", max_keys=25)
