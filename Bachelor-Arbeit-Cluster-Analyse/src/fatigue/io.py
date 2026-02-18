from pathlib import Path
import numpy as np
import pandas as pd
from scipy.io import loadmat

from .features import fatigue_features_from_timeseries

def load_parameters_R(path: Path) -> dict:
    data = loadmat(path, simplify_cells=True)
    return data["PARAMETERS"]["R"]

def build_fatigue_vector(mat_path: Path, whitelist: list[str], early_frac: float = 0.2) -> pd.Series:
    R = load_parameters_R(mat_path)

    out = {}
    for feat in whitelist:
        if feat not in R:
            continue
        ts = np.asarray(R[feat]).astype(float).ravel()
        f = fatigue_features_from_timeseries(ts, early_frac=early_frac)
        for k, v in f.items():
            out[f"{feat}_{k}"] = v

    if not out:
        raise ValueError("Whitelist produced empty vector. Check feature names.")
    return pd.Series(out)
