# diagnose_mat.py  – temporäres Diagnoseskript, gehört NICHT in den finalen Code
import sys
from pathlib import Path
import numpy as np
from scipy.io import loadmat

mat_path = r"C:\Users\Mika\Uni\BA\Beispiel Daten\P61_km01_0.mat"  # ← anpassen!

mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
mat.pop("__header__", None)
mat.pop("__version__", None)
mat.pop("__globals__", None)

print("=== TOP LEVEL KEYS ===")
print(list(mat.keys()))

# CONTACT untersuchen
contact = mat["CONTACT"]
trial_names = [k for k in contact.__dict__.keys() if not k.startswith("_")]
print("\n=== CONTACT TRIALS ===")
print(trial_names)

trial = getattr(contact, trial_names[0])
print("\n=== FELDER IM ERSTEN TRIAL ===")
print([k for k in trial.__dict__.keys() if not k.startswith("_")])

ct = getattr(trial, "ContactTimes")
ft = getattr(trial, "FlightTimes")
print("\n=== ContactTimes (erste 10 Werte) ===")
print(np.asarray(ct).flat[:10])
print("Anzahl:", len(np.asarray(ct).flat[:]))

print("\n=== FlightTimes (erste 10 Werte) ===")
print(np.asarray(ft).flat[:10])
print("Anzahl:", len(np.asarray(ft).flat[:]))

print("\n=== Verhältnis CT/FT (Mittelwert) ===")
ct_arr = np.asarray(ct, dtype=float).flatten()
ft_arr = np.asarray(ft, dtype=float).flatten()
print(f"Mean ContactTime: {ct_arr.mean():.4f} s")
print(f"Mean FlightTime:  {ft_arr.mean():.4f} s")
print(f"Mean Stride Time: {(ct_arr[:min(len(ct_arr),len(ft_arr))] + ft_arr[:min(len(ct_arr),len(ft_arr))]).mean():.4f} s")

# PARAMETERS untersuchen – Beinlänge suchen
print("\n=== PARAMETERS TOP KEYS ===")
params = mat.get("PARAMETERS", None)
if params is not None:
    print([k for k in params.__dict__.keys() if not k.startswith("_")])
    if hasattr(params, "R"):
        r = params.R
        print("\n=== PARAMETERS.R FELDER ===")
        fields = [k for k in r.__dict__.keys() if not k.startswith("_")]
        print(fields)
        # Nach Beinlänge suchen
        keywords = ["leg", "bein", "height", "groesse", "length", "anthropo", "body"]
        print("\n=== Beinlängen-Kandidaten ===")
        for f in fields:
            if any(kw in f.lower() for kw in keywords):
                val = getattr(r, f)
                print(f"  {f}: {val}")
else:
    print("PARAMETERS nicht gefunden")