from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MAT_PATH = ROOT / "Beispiel Daten" / "P61_km01_0.mat"

print("MAT_PATH:", MAT_PATH)
print("Exists:", MAT_PATH.exists())
print("Size (MB):", round(MAT_PATH.stat().st_size / 1e6, 2))

# Versuch 1: MAT v7.2 oder älter (scipy)
try:
    from scipy.io import loadmat
    data = loadmat(MAT_PATH, simplify_cells=True)
    print("\nLoaded with scipy.io.loadmat (MAT v7.2 or older)")
    print("Keys:", list(data.keys()))
    LOADER = "scipy"
except Exception as e:
    print("\nscipy loadmat failed:", e)
    data = None
    LOADER = None

# Versuch 2: MAT v7.3 (HDF5)
if data is None:
    try:
        import h5py
        with h5py.File(MAT_PATH, "r") as f:
            print("\nLoaded with h5py (MAT v7.3 / HDF5)")
            print("Keys:", list(f.keys()))
        LOADER = "h5py"
    except Exception as e:
        print("\nh5py failed:", e)
        raise RuntimeError("File is not a valid MATLAB .mat (neither scipy nor h5py readable)")

print("\n>>> Loader:", LOADER)
