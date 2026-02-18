import numpy as np
from scipy.stats import linregress

def fatigue_features_from_timeseries(x: np.ndarray, early_frac: float = 0.2) -> dict:
    x = np.asarray(x, dtype=float).ravel()
    n = len(x)
    if n < 5:
        raise ValueError(f"Need at least 5 windows, got {n}")

    k = max(1, int(n * early_frac))
    early = x[:k]
    late = x[-k:]

    delta = float(np.nanmean(late) - np.nanmean(early))
    t = np.arange(n, dtype=float)
    slope = float(linregress(t, x).slope)

    return {"delta": delta, "slope": slope}

