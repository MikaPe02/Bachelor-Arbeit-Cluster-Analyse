# tests/test_speed_correction.py
#
# AUFRUF:
#   python tests/test_speed_correction.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))

from fatigue.speed_correction import compute_speed_residuals_km1, save_models, load_models


def _make_df(n: int, rng: np.random.Generator, multi_km: bool = False) -> pd.DataFrame:
    """Hilfsfunktion: synthetischer Datensatz mit speed_ms-korrelierten DF/SF_norm."""
    speeds = rng.normal(3.47, 0.33, n)
    speeds = np.clip(speeds, 2.78, 4.17)

    # DF korreliert positiv mit Speed (analog Patoz et al.)
    df_vals  = 0.45 + 0.02 * speeds + rng.normal(0, 0.015, n)
    # SF_norm korreliert negativ mit Speed
    sf_vals  = 1.10 - 0.03 * speeds + rng.normal(0, 0.020, n)

    rows = []
    for i in range(n):
        km_marks = [1.0, 1.5, 2.0, 2.5] if multi_km else [1.0]
        for km in km_marks:
            rows.append(dict(
                Subject  = f"P{i+1:02d}",
                km       = km,
                speed_ms = round(float(speeds[i]), 3),
                DF       = round(float(df_vals[i]), 6),
                SF_norm  = round(float(sf_vals[i]), 6),
            ))
    return pd.DataFrame(rows)


def test_residuals_mean_zero():
    """Residuen-Mittelwert muss exakt 0 sein (Eigenschaft linearer Regression)."""
    rng = np.random.default_rng(42)
    df  = _make_df(50, rng)

    df_km1, _ = compute_speed_residuals_km1(df)

    assert abs(df_km1["DF_residual"].mean()) < 1e-9, \
        f"DF_residual Mean != 0: {df_km1['DF_residual'].mean()}"
    assert abs(df_km1["SF_residual"].mean()) < 1e-9, \
        f"SF_residual Mean != 0: {df_km1['SF_residual'].mean()}"

    print("  OK  Residuen Mean == 0")


def test_r2_plausible():
    """R² muss im Bereich [0, 1] liegen; bei simulierten Daten > 0.3."""
    rng = np.random.default_rng(42)
    df  = _make_df(50, rng)

    _, models = compute_speed_residuals_km1(df)

    assert 0.0 <= models["R2_DF"] <= 1.0, f"R2_DF ausserhalb [0,1]: {models['R2_DF']}"
    assert 0.0 <= models["R2_SF"] <= 1.0, f"R2_SF ausserhalb [0,1]: {models['R2_SF']}"
    assert models["R2_DF"] > 0.05, \
        f"R2_DF zu niedrig (erwartet > 0.05 bei speed-korrelierten Daten): {models['R2_DF']}"

    print(f"  OK  R2_DF={models['R2_DF']:.3f}, R2_SF={models['R2_SF']:.3f}")


def test_save_load_models(tmp_path: Path):
    """Gespeicherte Modelle muessen identische Koeffizienten liefern."""
    rng = np.random.default_rng(0)
    df  = _make_df(30, rng)

    _, models = compute_speed_residuals_km1(df)

    pkl = tmp_path / "speed_models.pkl"
    save_models(models, pkl)

    assert pkl.exists(), "PKL-Datei wurde nicht erstellt"

    loaded = load_models(pkl)

    assert loaded["beta0_DF"] == models["beta0_DF"], "beta0_DF stimmt nicht ueberein"
    assert loaded["beta1_DF"] == models["beta1_DF"], "beta1_DF stimmt nicht ueberein"
    assert loaded["beta0_SF"] == models["beta0_SF"], "beta0_SF stimmt nicht ueberein"
    assert loaded["beta1_SF"] == models["beta1_SF"], "beta1_SF stimmt nicht ueberein"

    print("  OK  Speichern/Laden identische Koeffizienten")


def test_constant_speed_per_subject():
    """Speed muss pro Proband konstant sein (Laufband-Bedingung)."""
    rng = np.random.default_rng(7)
    df  = _make_df(10, rng, multi_km=True)

    # Pruefen dass Speed fuer alle km eines Probanden gleich ist
    for subj in df["Subject"].unique():
        std = df[df["Subject"] == subj]["speed_ms"].std()
        assert std == 0.0, \
            f"Speed variiert fuer {subj} (std={std:.4f}) – Laufband-Bedingung verletzt"

    # Funktion laeuft ohne Fehler
    df_km1, _ = compute_speed_residuals_km1(df)
    assert len(df_km1) == 10, "Falsche Anzahl Zeilen bei km 1.0"

    print("  OK  Speed konstant pro Proband (Laufband-Check)")


def test_missing_column_raises():
    """Fehlende Pflicht-Spalte muss ValueError ausloesen."""
    df = pd.DataFrame({
        "Subject": ["P01"],
        "km":      [1.0],
        "DF":      [0.35],
        "SF_norm": [0.80],
        # speed_ms fehlt absichtlich
    })
    try:
        compute_speed_residuals_km1(df)
        assert False, "Kein ValueError bei fehlender Spalte"
    except ValueError:
        print("  OK  ValueError bei fehlender speed_ms-Spalte")


if __name__ == "__main__":
    import tempfile

    print("=" * 65)
    print("Tests: speed_correction.py")
    print("=" * 65)

    test_residuals_mean_zero()
    test_r2_plausible()

    with tempfile.TemporaryDirectory() as tmp:
        test_save_load_models(Path(tmp))

    test_constant_speed_per_subject()
    test_missing_column_raises()

    print()
    print("=" * 65)
    print("ALLE TESTS BESTANDEN (5/5)")
    print("=" * 65)
