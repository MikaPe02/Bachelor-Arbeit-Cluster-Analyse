from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]  # .../BA
SRC = ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"

print("SRC exists:", SRC.exists())
sys.path.insert(0, str(SRC))

import fatigue
print("fatigue imported from:", fatigue.__file__)
