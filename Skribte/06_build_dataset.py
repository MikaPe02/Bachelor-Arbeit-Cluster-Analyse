from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.insert(0, str(SRC))

from fatigue.io import build_fatigue_vector

DATA_DIR = ROOT / "Beispiel Daten"
WL_PATH = ROOT / "Skribte" / "whitelist.txt"

whitelist = [line.strip() for line in WL_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

rows = []
for mat_path in sorted(DATA_DIR.glob("*.mat")):
    subject_id = mat_path.stem.split("_")[0]  # z.B. P61 aus P61_km01_0
    vec = build_fatigue_vector(mat_path, whitelist, early_frac=0.2)
    vec.name = subject_id
    rows.append(vec)

X = pd.DataFrame(rows)
X.index.name = "subject"

out_csv = ROOT / "Skribte" / "fatigue_dataset.csv"
X.to_csv(out_csv)
print("Built dataset:", X.shape, "saved to", out_csv)
print("Missing values per column (top 10):")
print(X.isna().sum().sort_values(ascending=False).head(10))

