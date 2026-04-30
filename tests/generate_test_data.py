# tests/generate_test_data.py
#
# ZWECK: Synthetische Testdaten fuer die Clustering-Pipeline generieren.
#        Erzeugt 30 Probanden in 5 Gruppen nach van Oeveren et al. (2021).
#
# AUFRUF:
#   python tests/generate_test_data.py
#
# OUTPUT:
#   data/processed/dual_axis_dataset.csv  (ueberschrieben)
#   data/subjects.csv                      (ueberschrieben)

# ── Imports ──────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Bachelor-Arbeit-Cluster-Analyse" / "src"))

import config
from fatigue.features import estimate_leg_length
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED    = 42
N_PER_GROUP    = 6
KM_MARKS       = np.arange(1.0, 10.5, 0.5)   # 19 Punkte: 1.0, 1.5, ..., 10.0
NOISE_STD_DEFAULT = 0.008

# ── Gruppenparameter ──────────────────────────────────────────────────────────
#
# df_range / sf_range: gleichverteilter Startpunkt bei km 1.0
# df_slope / sf_slope: Gesamtveraenderung von km 1.0 bis km 10.0
# noise_std:           Standardabweichung des Rauschens pro km-Marke
#
# Quelle Bereiche: van Oeveren et al. (2021), Dual-Axis Framework
#
GROUPS = {
    "Stick": dict(
        df_range  = (0.68, 0.74),
        sf_range  = (0.72, 0.80),
        df_slope  = +0.02,
        sf_slope  =  0.00,
        noise_std = NOISE_STD_DEFAULT,
    ),
    "Bounce": dict(
        df_range  = (0.60, 0.66),
        sf_range  = (0.82, 0.90),
        df_slope  =  0.00,
        sf_slope  =  0.00,
        noise_std = 0.012,          # moderates Rauschen = leicht instabil
    ),
    "Push": dict(
        df_range  = (0.52, 0.58),
        sf_range  = (0.92, 1.00),
        df_slope  =  0.00,
        sf_slope  = -0.03,
        noise_std = NOISE_STD_DEFAULT,
    ),
    "Hop": dict(
        df_range  = (0.50, 0.56),
        sf_range  = (0.95, 1.03),
        df_slope  = +0.01,
        sf_slope  = -0.04,
        noise_std = NOISE_STD_DEFAULT,
    ),
    "Sit": dict(
        df_range  = (0.66, 0.72),
        sf_range  = (0.70, 0.78),
        df_slope  = +0.03,
        sf_slope  = -0.02,
        noise_std = NOISE_STD_DEFAULT,
    ),
}


def generate_dual_axis(rng: np.random.Generator) -> pd.DataFrame:
    """
    Erzeugt dual_axis_dataset.csv mit 30 Probanden x 19 km-Marken = 570 Zeilen.

    Pro Proband:
      - Startwert (km 1.0) uniform aus dem Gruppenbereich gezogen
      - Linearer Trend: value(km) = start + slope * (km - 1.0) / 9.0
      - Normalverteiltes Rauschen addiert
    """
    rows = []
    subject_nr = 1

    for true_cluster, (style, params) in enumerate(GROUPS.items(), start=1):
        for _ in range(N_PER_GROUP):
            subject_id = f"P{subject_nr:02d}"

            # Startwert bei km 1.0 – uniform im Gruppenbereich
            df_start = rng.uniform(*params["df_range"])
            sf_start = rng.uniform(*params["sf_range"])

            for km in KM_MARKS:
                # Linearer Trend normiert auf [0, 1] ueber die 9 km Strecke
                t = (km - 1.0) / 9.0

                df_val = (df_start + params["df_slope"] * t
                          + rng.normal(0.0, params["noise_std"]))
                sf_val = (sf_start + params["sf_slope"] * t
                          + rng.normal(0.0, params["noise_std"]))

                rows.append(dict(
                    Subject           = subject_id,
                    km                = km,
                    DF                = round(df_val, 6),
                    SF_norm           = round(sf_val, 6),
                    true_cluster      = true_cluster,
                    true_running_style= style,
                ))

            subject_nr += 1

    return pd.DataFrame(rows)


def generate_subjects(rng: np.random.Generator) -> pd.DataFrame:
    """
    Erzeugt subjects.csv mit 30 Probanden.

    body_height_m: N(1.75, 0.08), geclippt auf [1.55, 2.00]
    leg_length_m:  leer – damit estimate_leg_length() in der Pipeline greift
    speed_ms:      N(3.47, 0.33), geclippt auf [2.78, 4.17] (= 10-15 km/h)
    """
    subjects = [f"P{i:02d}" for i in range(1, 31)]
    heights  = rng.normal(1.75, 0.08, size=30)
    heights  = np.clip(heights, 1.55, 2.00).round(3)
    speeds   = rng.normal(3.47, 0.33, size=30)   # mu=12.5 km/h, sigma=1.2 km/h in m/s
    speeds   = np.clip(speeds, 2.78, 4.17).round(3)

    return pd.DataFrame(dict(
        Subject       = subjects,
        leg_length_m  = np.nan,          # bewusst leer – Schaetzung via body_height_m
        body_height_m = heights,
        speed_ms      = speeds,
        notes         = "synthetisch generiert",
    ))


def validate(df_dual: pd.DataFrame, df_subj: pd.DataFrame) -> None:
    """
    Validierungspruefungen nach der Erzeugung.

    1) Pflichtfelder und Zeilenzahlen
    2) estimate_leg_length()-Kompatibilitaet: leg_length_m leer, body_height_m gefuellt
    3) Gruppenstatistik bei km 1.0 – zeigt ob Gruppen klar getrennt sind
    """
    print("\n=== Validierung ===")

    # ── 1) Grundstruktur ─────────────────────────────────────────────────────
    assert set(["Subject", "km", "DF", "SF_norm",
                "true_cluster", "true_running_style"]).issubset(df_dual.columns), \
        "dual_axis_dataset: fehlende Spalten"
    assert len(df_dual) == 30 * 19, \
        f"Zeilenzahl falsch: {len(df_dual)} (erwartet 570)"
    assert df_dual["Subject"].nunique() == 30, \
        "Nicht 30 eindeutige Probanden"

    assert set(["Subject", "leg_length_m", "body_height_m", "speed_ms", "notes"]).issubset(df_subj.columns), \
        "subjects: fehlende Spalten"
    assert len(df_subj) == 30, f"subjects: {len(df_subj)} Zeilen (erwartet 30)"

    print("  OK  Zeilenzahlen und Spalten korrekt")

    # ── 2) estimate_leg_length()-Kompatibilitaet ─────────────────────────────
    n_empty_leg    = df_subj["leg_length_m"].isna().sum()
    n_filled_height= df_subj["body_height_m"].notna().sum()

    assert n_empty_leg == 30, \
        f"leg_length_m sollte ueberall leer sein, aber {30 - n_empty_leg} gefuellt"
    assert n_filled_height == 30, \
        f"body_height_m sollte ueberall gefuellt sein, aber {30 - n_filled_height} fehlen"

    # Beispielrechnung: zeigt was die Pipeline berechnen wuerde
    example_height = df_subj["body_height_m"].mean()
    example_leg    = estimate_leg_length(example_height)
    print(f"  OK  leg_length_m: alle {n_empty_leg} Eintraege leer")
    print(f"  OK  body_height_m: alle {n_filled_height} Eintraege gefuellt")
    print(f"      Beispiel: estimate_leg_length({example_height:.3f}m) = {example_leg:.4f}m")

    # ── 3) Gruppenstatistik bei km 1.0 ───────────────────────────────────────
    print("\n  Gruppenstatistik bei km 1.0 (Mittelwert +/- Std):")
    print(f"  {'Gruppe':<10}  {'DF mean':>8}  {'DF std':>7}  {'SF_norm mean':>13}  {'SF_norm std':>11}")
    print("  " + "-" * 57)

    df_km1 = df_dual[df_dual["km"] == 1.0]
    for style in GROUPS:
        mask = df_km1["true_running_style"] == style
        grp  = df_km1[mask]
        print(
            f"  {style:<10}  "
            f"{grp['DF'].mean():>8.4f}  "
            f"{grp['DF'].std():>7.4f}  "
            f"{grp['SF_norm'].mean():>13.4f}  "
            f"{grp['SF_norm'].std():>11.4f}"
        )

    print("\n  Alle Pruefungen bestanden.")


def main() -> None:
    print("=" * 55)
    print("Synthetische Testdaten generieren (seed={})".format(RANDOM_SEED))
    print("=" * 55)

    rng = np.random.default_rng(RANDOM_SEED)

    # ── Generieren ───────────────────────────────────────────────────────────
    print("\nGeneriere dual_axis_dataset.csv ...")
    df_dual = generate_dual_axis(rng)

    print("Generiere subjects.csv ...")
    df_subj = generate_subjects(rng)

    # ── Speed aus subjects in dual_axis mergen ───────────────────────────────
    df_dual = df_dual.merge(
        df_subj[["Subject", "speed_ms"]], on="Subject", how="left"
    )

    # ── Speichern ────────────────────────────────────────────────────────────
    config.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df_dual.to_csv(config.DUAL_AXIS_CSV, index=False)
    print(f"  -> {config.DUAL_AXIS_CSV}  ({len(df_dual)} Zeilen)")

    df_subj.to_csv(config.SUBJECTS_CSV, index=False)
    print(f"  -> {config.SUBJECTS_CSV}  ({len(df_subj)} Zeilen)")

    # ── Validierung: CSVs neu einlesen ───────────────────────────────────────
    df_dual_reload = pd.read_csv(config.DUAL_AXIS_CSV)
    df_subj_reload = pd.read_csv(config.SUBJECTS_CSV)

    validate(df_dual_reload, df_subj_reload)

    print("\n" + "=" * 55)
    print("Fertig. Naechster Schritt: python main.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
