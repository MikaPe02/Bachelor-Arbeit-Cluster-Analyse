# exploration/21_export_apa_tables.py
#
# ZWECK: Erstellt publikationsfertige APA-Tabellen aus den Pipeline-Outputs.
#        Liest automatisch die neuesten CSVs aus Outputs/Data/.
#
# OUTPUT:
#   Outputs/Data/apa_tabelle_deskriptiv_{suffix}.csv   — MW (SD) pro Cluster
#   Outputs/Data/apa_tabelle_anova_{suffix}.csv        — ANOVA-Ergebnisse
#
# AUFRUF:
#   python exploration/21_export_apa_tables.py

import sys
from pathlib import Path
import pandas as pd
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import config

DATA_DIR = config.OUTPUT_DATA_DIR


# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _find_latest(pattern: str) -> Path | None:
    """Findet die neueste Datei die dem Glob-Muster entspricht."""
    files = sorted(DATA_DIR.glob(pattern), key=lambda f: f.stat().st_mtime, reverse=True)
    return files[0] if files else None


def _format_p(p: float) -> str:
    """APA-konformes p-Format: < .001 oder .xxx"""
    if p < 0.001:
        return "< .001"
    return f".{round(p, 3):.3f}"[1:]  # entfernt führende 0


def _format_mw_sd(mw: float, sd: float, decimals: int = 3) -> str:
    fmt = f".{decimals}f"
    return f"{mw:{fmt}} ({sd:{fmt}})"


# ── Feature-Labels ────────────────────────────────────────────────────────────

FEATURE_LABELS = {
    "Delta_DF":  "Δ DF",
    "Slope_DF":  "Slope DF",
    "Delta_SF":  "Δ SF_norm",
    "Slope_SF":  "Slope SF_norm",
    "DF":        "Duty Factor",
    "SF_norm":   "SF_norm",
    "speed_ms":  "Speed (m/s)",
    "body_height_m": "Körpergröße (m)",
    "weight_kg": "Gewicht (kg)",
    "leg_length_m":  "Beinlänge (m)",
    "DF_start":  "DF Start (km 1)",
    "DF_end":    "DF Ende (km 10)",
    "SF_start":  "SF_norm Start (km 1)",
    "SF_end":    "SF_norm Ende (km 10)",
}

# Reihenfolge der Variablen in der deskriptiven Tabelle
DESKRIPTIV_ORDER = [
    "Duty_Factor_(DF)",
    "SF_norm",
    "Speed_m/s",
    "Körpergröße_m",
    "Gewicht_kg",
    "Beinlänge_m",
    "DF_start",
    "DF_end",
    "Delta_DF",
    "Slope_DF",
    "SF_start",
    "SF_end",
    "Delta_SF",
    "Slope_SF",
]

DESKRIPTIV_LABELS = {
    "Duty_Factor_(DF)": "Duty Factor",
    "SF_norm":          "SF_norm",
    "Speed_m/s":        "Speed (m/s)",
    "Körpergröße_m":    "Körpergröße (m)",
    "Gewicht_kg":       "Gewicht (kg)",
    "Beinlänge_m":      "Beinlänge (m)",
    "DF_start":         "DF Start (km 1)",
    "DF_end":           "DF Ende (km 10)",
    "Delta_DF":         "Δ DF",
    "Slope_DF":         "Slope DF",
    "SF_start":         "SF_norm Start (km 1)",
    "SF_end":           "SF_norm Ende (km 10)",
    "Delta_SF":         "Δ SF_norm",
    "Slope_SF":         "Slope SF_norm",
}

ANOVA_FEATURE_LABELS = {
    "Delta_DF":  "Δ DF",
    "Slope_DF":  "Slope DF",
    "Delta_SF":  "Δ SF_norm",
    "Slope_SF":  "Slope SF_norm",
    "DF":        "Duty Factor",
    "SF_norm":   "SF_norm",
    "speed_ms":  "Speed (m/s)",
    "body_height_m": "Körpergröße (m)",
    "weight_kg": "Gewicht (kg)",
    "leg_length_m":  "Beinlänge (m)",
}


# ── Tabelle 1: Deskriptive Statistik ─────────────────────────────────────────

def export_deskriptiv(path: Path) -> None:
    df = pd.read_csv(path)

    # Suffix aus Dateiname extrahieren
    suffix = path.stem.replace("descriptive_stats_clusters", "")

    cluster_cols = [c for c in df.columns if c == "Cluster"]
    clusters = df["Cluster"].tolist()
    n_per_cluster = dict(zip(df["Cluster"], df["N"]))

    # Spaltenheader: "Cluster 1 (n=21)"
    cluster_headers = [f"{c} (n={n_per_cluster[c]})" for c in clusters]

    rows = []
    for var_prefix in DESKRIPTIV_ORDER:
        mw_col = f"{var_prefix}_MW"
        sd_col = f"{var_prefix}_SD"

        # Flexibel suchen — Spaltennamen können leicht abweichen
        mw_candidates = [c for c in df.columns if c.startswith(var_prefix) and c.endswith("_MW")]
        sd_candidates = [c for c in df.columns if c.startswith(var_prefix) and c.endswith("_SD")]

        if not mw_candidates or not sd_candidates:
            continue

        mw_col = mw_candidates[0]
        sd_col = sd_candidates[0]

        label = DESKRIPTIV_LABELS.get(var_prefix, var_prefix)
        row = {"Variable": label}
        for i, cluster in enumerate(clusters):
            mw = df.loc[df["Cluster"] == cluster, mw_col].iloc[0]
            sd = df.loc[df["Cluster"] == cluster, sd_col].iloc[0]
            row[cluster_headers[i]] = _format_mw_sd(mw, sd)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"apa_tabelle_deskriptiv{suffix}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  -> Deskriptiv-Tabelle: {out_path}")
    print(out_df.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))


# ── Tabelle 2: ANOVA ─────────────────────────────────────────────────────────

def export_anova(path: Path) -> None:
    df = pd.read_csv(path)

    suffix = path.stem.replace("anova_results", "")

    # Freiheitsgrade: df_between = k-1, df_within = N-k
    # Aus Verfahren-Spalte k extrahieren
    verfahren = df["Verfahren"].iloc[0]
    k_match = re.search(r"k=(\d+)", verfahren)
    k = int(k_match.group(1)) if k_match else None

    # N aus deskriptiver CSV holen
    desc_path = _find_latest("descriptive_stats_clusters*.csv")
    N = None
    if desc_path is not None:
        desc = pd.read_csv(desc_path)
        N = int(desc["N"].sum())

    rows = []
    for _, row in df.iterrows():
        feat  = row["Feature"]
        label = ANOVA_FEATURE_LABELS.get(feat, feat)
        F     = row["F"]
        p     = row["p"]
        eta2  = row["eta2"]
        sig   = row["sig"]

        # df-Angabe
        if k is not None and N is not None:
            df_str = f"F({k-1}, {N-k})"
        else:
            df_str = "F"

        # Tukey: alle Paare signifikant?
        pair_sig_cols = [c for c in df.columns if c.endswith("_sig")]
        if pair_sig_cols:
            pair_results = [row[c] for c in pair_sig_cols if pd.notna(row[c])]
            alle_sig = all(v == "ja" for v in pair_results)
            tukey_str = "alle Paare sig." if alle_sig else "siehe Tukey-Tabelle"
        else:
            tukey_str = "—"

        rows.append({
            "Variable":    label,
            "Kategorie":   row["Kategorie"],
            df_str:        f"{F:.2f}",
            "p":           _format_p(p),
            "η²":          f"{eta2:.3f}",
            "Sig.":        sig,
            "Tukey HSD":   tukey_str,
        })

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"apa_tabelle_anova{suffix}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  -> ANOVA-Tabelle: {out_path}")
    print(out_df.to_string(index=False).encode("ascii", errors="replace").decode("ascii"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== APA-Tabellen Export ===\n")

    # Deskriptive Statistik
    desc_path = _find_latest("descriptive_stats_clusters*.csv")
    if desc_path:
        print(f"Lese: {desc_path.name}")
        export_deskriptiv(desc_path)
    else:
        print("WARNUNG: Keine descriptive_stats_clusters*.csv gefunden.")

    # ANOVA
    anova_path = _find_latest("anova_results*.csv")
    if anova_path:
        print(f"\nLese: {anova_path.name}")
        export_anova(anova_path)
    else:
        print("WARNUNG: Keine anova_results*.csv gefunden.")

    print("\n=== Fertig ===")
    print("Tipp: CSV in Google Sheets importieren, Tabelle markieren, in Google Docs einfuegen")


if __name__ == "__main__":
    main()
