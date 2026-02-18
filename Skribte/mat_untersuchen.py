from pathlib import Path
import numpy as np

path = Path(r"C:\Users\Mika\Uni\BA\Beispiel Daten\P61_km01_0.mat")

print("Existiert:", path.exists())
print("Dateigröße (MB):", round(path.stat().st_size / 1e6, 2))

# --- Versuch 1: scipy (MAT v7.2 oder älter)
try:
    from scipy.io import loadmat
    data = loadmat(path, simplify_cells=True)
    print("\nGeladen mit scipy.io.loadmat")
    print("Keys:", list(data.keys()))
    LOADER = "scipy"
except Exception as e:
    print("\nscipy loadmat fehlgeschlagen:", e)
    data = None
    LOADER = None

# --- Versuch 2: h5py (MAT v7.3)
if data is None:
    try:
        import h5py
        f = h5py.File(path, "r")
        print("\nGeladen mit h5py")
        print("Keys:", list(f.keys()))
        LOADER = "h5py"
    except Exception as e:
        print("\nAuch h5py fehlgeschlagen:", e)
        raise RuntimeError("Datei ist keine gültige MAT-Datei")

print("\n>>> Verwendeter Loader:", LOADER)
