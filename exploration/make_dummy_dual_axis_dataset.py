import numpy as np
import pandas as pd


def make_subject_ids(n: int) -> list[str]:
    return [f"P{61+i}" for i in range(n)]


def km_grid():
    # 1.0, 1.5, ..., 10.0
    return np.arange(1.0, 10.0 + 0.5, 0.5)


def generate_dummy_dual_axis(n_subjects: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    kms = km_grid()
    n_km = len(kms)

    subjects = make_subject_ids(n_subjects)

    rows = []
    patterns = ["stable", "df_up", "sf_down", "both", "df_down", "sf_up"]
    pattern_probs = np.array([0.20, 0.25, 0.25, 0.20, 0.05, 0.05])  # mostly the main 4

    for s in subjects:
        # Speed per subject (treadmill typical, just dummy values)
        speed = float(rng.uniform(2.6, 4.2))  # m/s (≈ 9.4–15.1 km/h)

        # baseline style
        DF0 = float(rng.uniform(0.35, 0.62))
        SF0 = float(rng.uniform(0.80, 1.10))

        pattern = rng.choice(patterns, p=pattern_probs)

        # fatigue amplitudes (small realistic-ish changes)
        amp_df = float(rng.uniform(0.00, 0.12))  # DF change over 10km
        amp_sf = float(rng.uniform(0.00, 0.18))  # SF_norm change over 10km

        # time normalization 0..1 across kms
        t = (kms - kms[0]) / (kms[-1] - kms[0])

        # base trends (linear) + slight nonlinearity + noise
        noise_df = rng.normal(0, 0.005, size=n_km)
        noise_sf = rng.normal(0, 0.008, size=n_km)

        # choose directions by pattern
        df_dir = 0.0
        sf_dir = 0.0

        if pattern == "stable":
            df_dir, sf_dir = 0.0, 0.0
            amp_df, amp_sf = 0.0, 0.0
        elif pattern == "df_up":
            df_dir, sf_dir = +1.0, 0.0
            amp_sf = 0.0
        elif pattern == "sf_down":
            df_dir, sf_dir = 0.0, -1.0
            amp_df = 0.0
        elif pattern == "both":
            df_dir, sf_dir = +1.0, -1.0
        elif pattern == "df_down":
            df_dir, sf_dir = -1.0, 0.0
            amp_sf = 0.0
        elif pattern == "sf_up":
            df_dir, sf_dir = 0.0, +1.0
            amp_df = 0.0

        # slight curvature: fatigue often accelerates a bit
        curve = t**1.3

        DF = DF0 + df_dir * amp_df * curve + noise_df
        SF = SF0 + sf_dir * amp_sf * curve + noise_sf

        # clamp to plausible ranges
        DF = np.clip(DF, 0.25, 0.75)
        SF = np.clip(SF, 0.60, 1.30)

        for km, df_val, sf_val in zip(kms, DF, SF):
            rows.append(
                {
                    "Subject": s,
                    "km": float(km),
                    "DF": float(df_val),
                    "SF_norm": float(sf_val),
                    "Speed_mps": speed,
                    "Pattern": pattern,  # nur Debug/Validierung
                }
            )

    return pd.DataFrame(rows)


def main():
    df = generate_dummy_dual_axis(n_subjects=20, seed=42)

    # Für deine Pipeline: wir speichern ohne Pattern-Spalte (damit es “real” ist)
    df_out = df.drop(columns=["Pattern"])

    df_out.to_csv("dual_axis_dataset.txt", sep="\t", index=False)

    # Zusätzlich Debug-Datei mit Pattern (zum Prüfen ob Clustering grob passt)
    df.to_csv("dual_axis_dataset_debug.txt", sep="\t", index=False)

    print("Created:")
    print("- dual_axis_dataset.txt")
    print("- dual_axis_dataset_debug.txt")
    print("\nPreview:")
    print(df_out.head(10))


if __name__ == "__main__":
    main()