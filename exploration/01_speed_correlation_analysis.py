# exploration/01_speed_correlation_analysis.py
#
# ZIEL: Entscheiden ob Residual-Bereinigung notwendig ist basierend auf
#       Korrelation zwischen Speed und DF/SF_norm bei km 1.

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

# ── Pfade ─────────────────────────────────────────────────────────────────────
DUAL_AXIS_CSV = PROJECT_ROOT / "Input" / "processed" / "dual_axis_dataset.csv"
SUBJECTS_CSV  = PROJECT_ROOT / "Input" / "subjects.csv"
OUTPUT_PLOTS  = PROJECT_ROOT / "Outputs" / "Plots"
OUTPUT_DATA   = PROJECT_ROOT / "Outputs" / "Data"


def _sig_stars(p: float) -> str:
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


def _interpretation(r: float) -> str:
    if abs(r) >= 0.5: return "Stark"
    if abs(r) >= 0.4: return "Moderat"
    return "Schwach"


def load_data() -> pd.DataFrame:
    for path, name in [(DUAL_AXIS_CSV, "dual_axis_dataset.csv"), (SUBJECTS_CSV, "subjects.csv")]:
        if not path.exists():
            raise SystemExit(f"FEHLER: {name} nicht gefunden: {path}\n"
                             "Bitte zuerst extract_to_csv.py ausführen.")

    df = pd.read_csv(DUAL_AXIS_CSV)
    subjects = pd.read_csv(SUBJECTS_CSV, usecols=["Subject", "speed_ms"])

    if "speed_ms" not in df.columns:
        df = df.merge(subjects, on="Subject", how="left")

    required = {"Subject", "km", "DF", "SF_norm", "speed_ms"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"FEHLER: Fehlende Spalten: {missing}")

    df = df[df["km"].between(0.999, 1.001)].copy()
    df["speed_kmh"] = df["speed_ms"] * 3.6

    if len(df) < 30:
        print(f"  WARNUNG: Nur {len(df)} Läufer bei km 1 (empfohlen: ≥ 30)")

    return df


def descriptive_stats(df: pd.DataFrame) -> None:
    print("\n  DATENSATZ:")
    print(f"    Läufer bei km 1:   {len(df)}")
    for col, label in [("speed_kmh", "Speed [km/h]"), ("DF", "DF"), ("SF_norm", "SF_norm")]:
        s = df[col]
        print(f"    {label+'-Range:':<18} {s.min():.2f} - {s.max():.2f}  "
              f"(Mean: {s.mean():.2f} ± {s.std():.2f})")


def correlations(df: pd.DataFrame) -> tuple[float, float, float, float]:
    r_df, p_df   = stats.pearsonr(df["speed_kmh"], df["DF"])
    r_sf, p_sf   = stats.pearsonr(df["speed_kmh"], df["SF_norm"])
    return r_df, p_df, r_sf, p_sf


def regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2    = float(model.score(x.reshape(-1, 1), y))
    b0    = float(model.intercept_)
    b1    = float(model.coef_[0])
    return r2, b0, b1


def decide(r_df: float, r_sf: float) -> tuple[str, str, str]:
    max_r = max(abs(r_df), abs(r_sf))
    if max_r < 0.4:
        return "A", "OPTION A: Mixed Model (Clustering auf Rohdaten)", \
               f"Schwache Korrelation (max |r| = {max_r:.3f} < 0.4) -> Speed-Einfluss gering"
    if max_r >= 0.5:
        return "B", "OPTION B: Residual-Bereinigung erforderlich", \
               f"Starke Korrelation (max |r| = {max_r:.3f} >= 0.5) -> Ohne Bereinigung clustert ihr Speed-Gruppen!"
    return "UNCLEAR", "GRAUZONE: Mit Professor diskutieren", \
           f"Moderate Korrelation (max |r| = {max_r:.3f}, 0.4 <= r < 0.5) -> Beide Optionen vertretbar"


_RCPARAMS = {
    "font.size": 11, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10,
}

_PLOTS = [
    ("DF",     "#4477AA", "Duty Factor [–]",              "speed_correlation_DF.png"),
    ("SF_norm","#228833", "Normierte Schrittfrequenz [–]", "speed_correlation_SF_norm.png"),
]


def make_plot(df, r_df, p_df, r2_df, b0_df, b1_df,
              r_sf, p_sf, r2_sf, b0_sf, b1_sf, decision_text) -> list[Path]:
    plt.rcParams.update(_RCPARAMS)
    OUTPUT_PLOTS.mkdir(parents=True, exist_ok=True)

    x_range = np.linspace(df["speed_kmh"].min(), df["speed_kmh"].max(), 100)
    saved = []

    for (y_col, color, ylabel, fname), (r, p, r2, b0, b1) in zip(
        _PLOTS,
        [(r_df, p_df, r2_df, b0_df, b1_df), (r_sf, p_sf, r2_sf, b0_sf, b1_sf)],
    ):
        fig, ax = plt.subplots(figsize=(16 / 2.54, 13 / 2.54))

        ax.scatter(df["speed_kmh"], df[y_col], color=color, alpha=0.65, s=40, zorder=3)
        ax.plot(x_range, b0 + b1 * x_range, color="red", linewidth=1.5,
                label=f"Regression (r = {r:.3f}, R² = {r2:.3f}, p < .001)")

        ax.set_xlabel("Laufgeschwindigkeit [km/h]")
        ax.set_ylabel(ylabel)
        ax.legend(frameon=True, fontsize=10)
        ax.grid(True, linewidth=0.5, alpha=0.5)
        for spine in ax.spines.values():
            spine.set_visible(True)

        fig.tight_layout()
        out = OUTPUT_PLOTS / fname
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out)
        print(f"  -> Gespeichert: {out}")

    return saved


def save_results(r_df, p_df, r2_df, r_sf, p_sf, r2_sf, code, decision_text, reason):
    OUTPUT_DATA.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame([
        {"Variable": "DF",     "r": r_df, "p_value": p_df, "R2": r2_df, "Interpretation": _interpretation(r_df)},
        {"Variable": "SF_norm","r": r_sf, "p_value": p_sf, "R2": r2_sf, "Interpretation": _interpretation(r_sf)},
    ])
    csv_path = OUTPUT_DATA / "speed_correlation_summary.csv"
    summary.to_csv(csv_path, index=False)

    txt_path = OUTPUT_DATA / "method_decision.txt"
    txt_path.write_text(f"{code}\n{decision_text}\n{reason}\n", encoding="utf-8")

    return csv_path, txt_path


def main():
    print("=" * 70)
    print("SPEED-KORRELATIONSANALYSE - METHODENENTSCHEIDUNG")
    print("=" * 70)

    df = load_data()
    descriptive_stats(df)

    r_df, p_df, r_sf, p_sf = correlations(df)

    x = df["speed_kmh"].values
    r2_df, b0_df, b1_df = regression(x, df["DF"].values)
    r2_sf, b0_sf, b1_sf = regression(x, df["SF_norm"].values)

    code, decision_text, reason = decide(r_df, r_sf)
    max_r = max(abs(r_df), abs(r_sf))

    print("\n  KORRELATIONEN:")
    print(f"    Speed ~ DF:        r = {r_df:+.3f}, p = {p_df:.4f} {_sig_stars(p_df)}")
    print(f"    Speed ~ SF_norm:   r = {r_sf:+.3f}, p = {p_sf:.4f} {_sig_stars(p_sf)}")

    print("\n  REGRESSIONSMODELLE:")
    print(f"    DF ~ Speed:        R² = {r2_df:.3f} ({r2_df*100:.1f}% erklärt durch Speed)")
    print(f"                       DF = {b0_df:.4f} + ({b1_df:.4f}) × speed_kmh")
    print(f"    SF ~ Speed:        R² = {r2_sf:.3f} ({r2_sf*100:.1f}% erklärt durch Speed)")
    print(f"                       SF = {b0_sf:.4f} + ({b1_sf:.4f}) × speed_kmh")

    print("\n" + "=" * 70)
    print(f"  ENTSCHEIDUNG: {decision_text}")
    print(f"  BEGRÜNDUNG:   {reason}")
    print("=" * 70)

    print(f"\n  INTERPRETATION:")
    print(f"    Speed erklärt {r2_df*100:.1f}% der DF-Varianz und {r2_sf*100:.1f}% der SF-Varianz.")
    if code == "B":
        print("    OHNE Residual-Bereinigung würde Clustering Speed-Gruppen finden,")
        print("    nicht Laufstil-Gruppen!")
        print("\n    EMPFEHLUNG: Implementiere Speed-Bereinigung (Residual-Methode)")
    elif code == "A":
        print("    Speed-Einfluss gering → Clustering direkt auf DF und SF_norm möglich.")
    else:
        print("    Grenzfall → Entscheidung mit Betreuer absprechen.")

    make_plot(df, r_df, p_df, r2_df, b0_df, b1_df,
              r_sf, p_sf, r2_sf, b0_sf, b1_sf, decision_text)
    csv_path, txt_path = save_results(r_df, p_df, r2_df, r_sf, p_sf, r2_sf,
                                      code, decision_text, reason)

    print("\n" + "=" * 70)
    print("  Ergebnisse gespeichert:")
    print(f"    Outputs/Plots/speed_correlation_DF.png")
    print(f"    Outputs/Plots/speed_correlation_SF_norm.png")
    print(f"    Outputs/Data/speed_correlation_summary.csv")
    print(f"    Outputs/Data/method_decision.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
