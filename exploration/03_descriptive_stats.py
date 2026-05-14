# exploration/03_descriptive_stats.py
#
# ZIEL: Deskriptive Beschreibung der Stichprobe.
#
#   Block 1 – Stichprobe (Input/subjects.csv):
#     N gesamt, M/W, dominantes Bein,
#     body_height_m, weight_kg, leg_length_m, speed_ms
#
#   Block 2 – Biomechanik bei km 1.0 (Input/processed/dual_axis_dataset.csv):
#     DF, SF_norm
#
# OUTPUT: Outputs/Data/descriptive_stats_sample.csv
#
# VORAUSSETZUNG:
#   - Input/subjects.csv
#   - Input/processed/dual_axis_dataset.csv

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

SUBJECTS_CSV  = PROJECT_ROOT / "Input" / "subjects.csv"
DUAL_AXIS_CSV = PROJECT_ROOT / "Input" / "processed" / "dual_axis_dataset.csv"
OUTPUT_DATA   = PROJECT_ROOT / "Outputs" / "Data"


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _stats_row(series: pd.Series, decimals: int = 3) -> dict:
    s = series.dropna()
    return {
        "N":   len(s),
        "MW":  round(s.mean(), decimals),
        "SD":  round(s.std(ddof=1), decimals),
        "Min": round(s.min(), decimals),
        "Max": round(s.max(), decimals),
    }


def _print_table(title: str, rows: list[tuple[str, dict]]) -> None:
    col_w = {"Variable": 22, "N": 5, "MW": 9, "SD": 9, "Min": 9, "Max": 9}
    header = (
        f"  {'Variable':<{col_w['Variable']}}"
        f"{'N':>{col_w['N']}}"
        f"{'MW':>{col_w['MW']}}"
        f"{'SD':>{col_w['SD']}}"
        f"{'Min':>{col_w['Min']}}"
        f"{'Max':>{col_w['Max']}}"
    )
    sep = "  " + "-" * sum(col_w.values())
    print(f"\n{title}")
    print(sep)
    print(header)
    print(sep)
    for label, s in rows:
        print(
            f"  {label:<{col_w['Variable']}}"
            f"{s['N']:>{col_w['N']}}"
            f"{s['MW']:>{col_w['MW']}.3f}"
            f"{s['SD']:>{col_w['SD']}.3f}"
            f"{s['Min']:>{col_w['Min']}.3f}"
            f"{s['Max']:>{col_w['Max']}.3f}"
        )
    print(sep)


# ── Laden ─────────────────────────────────────────────────────────────────────

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    for path, name in [(SUBJECTS_CSV, "subjects.csv"), (DUAL_AXIS_CSV, "dual_axis_dataset.csv")]:
        if not path.exists():
            raise SystemExit(
                f"FEHLER: {name} nicht gefunden: {path}\n"
                "Bitte zuerst extract_to_csv.py ausfuehren."
            )
    subjects = pd.read_csv(SUBJECTS_CSV)
    dual     = pd.read_csv(DUAL_AXIS_CSV)
    km1      = dual[np.abs(dual["km"] - 1.0) <= 1e-6].copy()
    return subjects, km1


# ── Block 1: Stichprobenbeschreibung ─────────────────────────────────────────

def block1_sample(subjects: pd.DataFrame) -> pd.DataFrame:
    n_total    = len(subjects)
    sex_counts = subjects["sex"].str.upper().value_counts()
    n_m = sex_counts.get("M", 0)
    n_w = sex_counts.get("W", 0)

    print("\n" + "=" * 65)
    print("BLOCK 1 – STICHPROBENBESCHREIBUNG")
    print("=" * 65)
    print(f"\n  N gesamt:       {n_total}  (Männlich: {n_m}, Weiblich: {n_w})")

    dom = subjects["dominant_leg"].str.strip().str.lower().value_counts()
    dom_str = "  ".join(f"{k.capitalize()}: {v}" for k, v in dom.items())
    print(f"  Dominantes Bein: {dom_str}")

    vars_ = [
        ("Körpergröße [m]", subjects["body_height_m"]),
        ("Gewicht [kg]",    subjects["weight_kg"]),
        ("Beinlänge [m]",   subjects["leg_length_m"]),
        ("Speed [m/s]",     subjects["speed_ms"]),
    ]
    if "SF_hz_km1" in subjects.columns:
        vars_.append(("SF bei km 1.0 [Hz]", subjects["SF_hz_km1"]))
    rows = [(lbl, _stats_row(s)) for lbl, s in vars_]
    _print_table("  Kontinuierliche Variablen (MW ± SD, Min–Max):", rows)

    records = [{"Gruppe": "Gesamt", "Variable": lbl, **stats} for lbl, stats in rows]
    return pd.DataFrame(records)


# ── Block 2: Biomechanik bei km 1.0 ──────────────────────────────────────────

def block2_biomechanics(km1: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("BLOCK 2 – BIOMECHANIK BEI km 1.0  (Ausgangszustand)")
    print("=" * 65)

    vars_ = [
        ("Duty Factor (DF)", km1["DF"]),
        ("SF_norm",          km1["SF_norm"]),
    ]
    rows = [(lbl, _stats_row(s)) for lbl, s in vars_]
    _print_table("  Alle Probanden (MW ± SD, Min–Max):", rows)

    records = [{"Gruppe": "Gesamt km1.0", "Variable": lbl, **stats} for lbl, stats in rows]
    return pd.DataFrame(records)


# ── Speichern ─────────────────────────────────────────────────────────────────

def save_results(df1: pd.DataFrame, df2: pd.DataFrame) -> None:
    combined = pd.concat([df1, df2], ignore_index=True)
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DATA / "descriptive_stats_sample.csv"
    combined.to_csv(out, index=False)
    print(f"\n  -> Gespeichert: {out.relative_to(PROJECT_ROOT)}")


# ── Einstiegspunkt ────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("DESKRIPTIVE STATISTIK – STICHPROBE")
    print("=" * 65)

    subjects, km1 = load_data()
    df1 = block1_sample(subjects)
    df2 = block2_biomechanics(km1)
    save_results(df1, df2)

    print("\n" + "=" * 65)
    print("Fertig.")
    print("=" * 65)


if __name__ == "__main__":
    main()
