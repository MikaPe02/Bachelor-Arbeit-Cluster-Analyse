# exploration/24_lmm_verlauf.py
#
# ZWECK: Mixed Linear Model (LMM) fuer den Verlauf von DF und SF_norm ueber km.
#        Testet ob sich die Cluster signifikant im Verlauf unterscheiden.
#
# MODELL:
#   Variable ~ Cluster + km + Cluster:km + (1|Subject)
#
#   Cluster        = Haupteffekt: liegen Cluster generell verschieden hoch?
#   km             = Haupteffekt: veraendert sich die Variable ueber km (global)?
#   Cluster:km     = Interaktion: unterscheiden sich die Steigungen der Cluster?
#                    -> Das ist die zentrale Frage fuer Verlaufsunterschiede
#   (1|Subject)    = Random Effect: jeder Proband hat seinen eigenen Intercept
#                    (beruecksichtigt dass km-Messungen pro Proband korreliert sind)
#
# INTERPRETATION:
#   Signifikante Interaktion Cluster:km -> Cluster unterscheiden sich im Verlauf
#   Nicht-signifikante Interaktion      -> Cluster verlaufen parallel (nur verschoben)
#
# AUFRUF:
#   python exploration/24_lmm_verlauf.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))
import config

# ── Hilfsfunktionen ──────────────────────────────────────────────────────────

def _find_latest(pattern: str) -> Path | None:
    files = sorted(
        config.OUTPUT_DATA_DIR.glob(pattern),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    return files[0] if files else None


def _format_p(p: float) -> str:
    if p < 0.001:
        return "< .001"
    return f".{round(p, 3):.3f}"[1:]


# ── LMM ──────────────────────────────────────────────────────────────────────

def run_lmm(df: pd.DataFrame, feature: str, feature_label: str) -> dict:
    """
    Fuehrt LMM fuer eine Variable durch und gibt Ergebnisse zurueck.

    Parameters
    ----------
    df            : DataFrame mit Spalten Subject, km, cluster_label, <feature>
    feature       : Spaltenname der abhaengigen Variable
    feature_label : Anzeigename fuer Terminal-Output
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  FEHLER: statsmodels nicht installiert.")
        print("  Loesung: pip install statsmodels")
        return {}

    df = df[["Subject", "km", "cluster_label", feature]].dropna().copy()
    df["cluster_label"] = df["cluster_label"].astype(str)
    df["km_centered"]   = df["km"] - df["km"].mean()

    # Referenz-Cluster: erster in sortierter Reihenfolge
    cluster_ids = sorted(df["cluster_label"].unique())
    ref         = cluster_ids[0]
    df["cluster_label"] = pd.Categorical(df["cluster_label"], categories=cluster_ids)

    print(f"\n{'='*60}")
    print(f"  LMM: {feature_label}")
    print(f"{'='*60}")
    print(f"  Referenz-Cluster: {ref}")
    print(f"  N Probanden     : {df['Subject'].nunique()}")
    print(f"  N Beobachtungen : {len(df)}")

    formula = f"{feature} ~ cluster_label * km_centered"

    try:
        model  = smf.mixedlm(formula, df, groups=df["Subject"])
        result = model.fit(reml=True)
    except Exception as e:
        print(f"  FEHLER beim Modell-Fit: {e}")
        return {}

    print(f"\n  Modell: {feature} ~ Cluster + km + Cluster:km + (1|Subject)")
    print(f"  Log-Likelihood: {result.llf:.2f}")
    print()

    # Ergebnistabelle
    col_w = {"Term": 40, "Koeff": 10, "SE": 10, "z": 8, "p": 10, "sig": 5}
    header = (
        f"  {'Term':<{col_w['Term']}}"
        f"{'Koeff':>{col_w['Koeff']}}"
        f"{'SE':>{col_w['SE']}}"
        f"{'z':>{col_w['z']}}"
        f"{'p':>{col_w['p']}}"
        f"{'sig':>{col_w['sig']}}"
    )
    sep = "  " + "-" * sum(col_w.values())
    print(sep)
    print(header)
    print(sep)

    records = []
    for term in result.params.index:
        coef  = result.params[term]
        se    = result.bse[term]
        z     = result.tvalues[term]
        p     = result.pvalues[term]
        sig   = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))

        # Interaktionsterme hervorheben
        is_interaction = ":" in term
        marker = " <-- Interaktion" if is_interaction else ""

        print(
            f"  {term:<{col_w['Term']}}"
            f"{coef:>{col_w['Koeff']}.4f}"
            f"{se:>{col_w['SE']}.4f}"
            f"{z:>{col_w['z']}.3f}"
            f"{p:>{col_w['p']}.4f}"
            f"  {sig}{marker}"
        )
        records.append({
            "Feature":      feature,
            "Term":         term,
            "Koeffizient":  round(coef, 5),
            "SE":           round(se, 5),
            "z":            round(z, 3),
            "p":            round(p, 4),
            "sig":          sig,
            "Interaktion":  "ja" if is_interaction else "nein",
        })

    print(sep)

    # Interpretation der Interaktion
    interaction_terms = [r for r in records if r["Interaktion"] == "ja"]
    sig_interactions  = [r for r in interaction_terms if r["p"] < 0.05]

    print()
    if sig_interactions:
        print(f"  ERGEBNIS: Signifikante Verlaufsunterschiede zwischen Clustern (Interaktion Cluster:km)")
        for r in sig_interactions:
            cluster = r["Term"].replace("cluster_label[T.", "").replace("]:km_centered", "")
            richtung = "staerker steigend" if r["Koeffizient"] > 0 else "staerker sinkend"
            print(f"    Cluster {cluster} vs. {ref}: {richtung} (beta={r['Koeffizient']:+.4f}, p={_format_p(r['p'])})")
    else:
        print(f"  ERGEBNIS: Keine signifikanten Verlaufsunterschiede (parallele Verlaeufe)")
        print(f"  -> Cluster unterscheiden sich im Niveau aber nicht in der Steigung")

    return {"feature": feature, "records": records, "n_sig_interactions": len(sig_interactions)}


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== Mixed Linear Model: Verlaufsanalyse ===\n")

    # Daten laden
    if not config.DUAL_AXIS_CSV.exists():
        print(f"FEHLER: {config.DUAL_AXIS_CSV} nicht gefunden.")
        return

    df_long = pd.read_csv(config.DUAL_AXIS_CSV)

    labels_path = _find_latest("cluster_labels*.csv")
    if labels_path is None:
        print("FEHLER: Keine cluster_labels*.csv gefunden. Bitte zuerst main.py ausfuehren.")
        return

    print(f"Cluster-Labels: {labels_path.name}")
    df_labels = pd.read_csv(labels_path)

    label_col = "cluster_label"
    if label_col not in df_labels.columns:
        cols = [c for c in df_labels.columns if c != "Subject"]
        df_labels = df_labels.rename(columns={cols[0]: label_col})

    # Zusammenfuehren
    df = df_long.merge(df_labels[["Subject", label_col]], on="Subject", how="inner")
    df = df.rename(columns={" Subject": "Subject"}) if " Subject" in df.columns else df

    suffix = labels_path.stem.replace("cluster_labels", "")

    all_records = []

    # LMM fuer DF
    res_df = run_lmm(df, "DF", "Duty Factor")
    if res_df:
        all_records.extend(res_df["records"])

    # LMM fuer SF_norm
    res_sf = run_lmm(df, "SF_norm", "Normierte Schrittfrequenz")
    if res_sf:
        all_records.extend(res_sf["records"])

    # CSV speichern
    if all_records:
        config.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = config.OUTPUT_DATA_DIR / f"lmm_results{suffix}.csv"
        pd.DataFrame(all_records).to_csv(out, index=False)
        print(f"\n  -> LMM-Ergebnisse gespeichert: {out}")

    print("\n=== Fertig ===")
    print("Hinweis: Signifikante Cluster:km Interaktion = Cluster unterscheiden sich im Verlauf")
    print("         Nicht-signifikant = Cluster verlaufen parallel, unterscheiden sich nur im Niveau")


if __name__ == "__main__":
    main()
