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
    "Delta_DF":       "Δ DF",
    "Slope_DF":       "Slope DF (gesamt)",
    "Slope_DF_early": "Slope DF (km 1-5)",
    "Slope_DF_late":  "Slope DF (km 6-10)",
    "Delta_SF":       "Δ SF_norm",
    "Slope_SF":       "Slope SF_norm (gesamt)",
    "Slope_SF_early": "Slope SF_norm (km 1-5)",
    "Slope_SF_late":  "Slope SF_norm (km 6-10)",
    "DF":             "Duty Factor",
    "SF_norm":        "SF_norm",
    "speed_ms":       "Speed (m/s)",
    "body_height_m":  "Körpergröße (m)",
    "weight_kg":      "Gewicht (kg)",
    "leg_length_m":   "Beinlänge (m)",
}


# Variablen-Reihenfolge fuer Fatigue-Deskriptiv
FATIGUE_DESKRIPTIV_ORDER = [
    "DF_Start_(km_1)",
    "SF_norm_Start_(km_1)",
    "Speed_m",
    "Koerpergroesse_m",
    "Gewicht_kg",
    "Beinlaenge_m",
    "Delta_DF",
    "Slope_DF",
    "Slope_DF_early",
    "Slope_DF_late",
    "Delta_SF",
    "Slope_SF",
    "Slope_SF_early",
    "Slope_SF_late",
]

FATIGUE_DESKRIPTIV_LABELS = {
    "DF_Start_(km_1)":      "DF Start (km 1)",
    "SF_norm_Start_(km_1)": "SF_norm Start (km 1)",
    "Speed_m":              "Laufgeschwindigkeit (m/s)",
    "Koerpergroesse_m":     "Körpergröße (m)",
    "Gewicht_kg":           "Gewicht (kg)",
    "Beinlaenge_m":         "Beinlänge (m)",
    "Delta_DF":             "Δ DF",
    "Slope_DF":             "Slope DF (gesamt)",
    "Slope_DF_early":       "Slope DF (km 1–5)",
    "Slope_DF_late":        "Slope DF (km 6–10)",
    "Delta_SF":             "Δ SF_norm",
    "Slope_SF":             "Slope SF_norm (gesamt)",
    "Slope_SF_early":       "Slope SF_norm (km 1–5)",
    "Slope_SF_late":        "Slope SF_norm (km 6–10)",
}


# ── Tabelle 1: Deskriptive Statistik ─────────────────────────────────────────

def _export_deskriptiv_generic(path: Path, var_order: list[str], var_labels: dict, stem_strip: str, out_prefix: str) -> None:
    df = pd.read_csv(path)
    suffix = path.stem.replace(stem_strip, "")
    clusters = df["Cluster"].tolist()
    n_per_cluster = dict(zip(df["Cluster"], df["N"]))
    cluster_headers = [f"{c} (n={n_per_cluster[c]})" for c in clusters]

    rows = []
    for var_prefix in var_order:
        mw_candidates = [c for c in df.columns if c.startswith(var_prefix) and c.endswith("_MW")]
        sd_candidates = [c for c in df.columns if c.startswith(var_prefix) and c.endswith("_SD")]
        if not mw_candidates or not sd_candidates:
            continue
        mw_col = mw_candidates[0]
        sd_col = sd_candidates[0]
        label = var_labels.get(var_prefix, var_prefix)
        row = {"Variable": label}
        for i, cluster in enumerate(clusters):
            mw = df.loc[df["Cluster"] == cluster, mw_col].iloc[0]
            sd = df.loc[df["Cluster"] == cluster, sd_col].iloc[0]
            row[cluster_headers[i]] = _format_mw_sd(mw, sd)
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"{out_prefix}{suffix}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  -> Deskriptiv-Tabelle: {out_path}")
    print(out_df.to_string(index=False).encode("cp1252", errors="replace").decode("cp1252"))


def export_deskriptiv(path: Path) -> None:
    _export_deskriptiv_generic(path, DESKRIPTIV_ORDER, DESKRIPTIV_LABELS,
                               "descriptive_stats_clusters", "apa_tabelle_deskriptiv_stil")


def export_deskriptiv_fatigue(path: Path) -> None:
    _export_deskriptiv_generic(path, FATIGUE_DESKRIPTIV_ORDER, FATIGUE_DESKRIPTIV_LABELS,
                               "descriptive_stats_fatigue_clusters", "apa_tabelle_deskriptiv_fatigue")


# ── Tabelle 2: ANOVA ─────────────────────────────────────────────────────────

def export_anova(path: Path, desc_pattern: str = "descriptive_stats_clusters*.csv", out_prefix: str = "apa_tabelle_anova_stil") -> None:
    df = pd.read_csv(path)

    suffix = path.stem.replace("anova_results", "")

    # Freiheitsgrade: df_between = k-1, df_within = N-k
    verfahren = df["Verfahren"].iloc[0]
    k_match = re.search(r"k=(\d+)", verfahren)
    k = int(k_match.group(1)) if k_match else None

    # N aus passender deskriptiver CSV holen
    desc_path = _find_latest(desc_pattern)
    N = None
    if desc_path is not None:
        desc = pd.read_csv(desc_path)
        N = int(desc["N"].sum())

    # Alle Cluster-Paare aus Spaltennamen ermitteln (Reihenfolge aus erster Zeile)
    d_cols    = [c for c in df.columns if c.endswith("_cohens_d")]
    sig_cols  = [c for c in df.columns if c.endswith("_sig") and not c.endswith("_cohens_d")]
    padj_cols = [c for c in df.columns if c.endswith("_p_adj")]

    # Paare in Reihenfolge aus d_cols extrahieren
    pairs = [c.replace("_cohens_d", "") for c in d_cols]

    rows = []
    for _, row in df.iterrows():
        feat  = row["Feature"]
        label = ANOVA_FEATURE_LABELS.get(feat, feat)
        F     = row["F"]
        p     = row["p"]
        eta2  = row["eta2"]
        sig   = row["sig"]

        if k is not None and N is not None:
            df_str = f"F({k-1}, {N-k})"
        else:
            df_str = "F"

        out_row = {
            "Variable":  label,
            "Kategorie": row["Kategorie"],
            df_str:      f"{F:.2f}",
            "p":         _format_p(p),
            "eta2":      f"{eta2:.3f}",
            "Sig.":      sig,
        }

        # Pro Paar: p_adj + Cohen's d als eigene Spalten
        for pair in pairs:
            padj_col = f"{pair}_p_adj"
            d_col    = f"{pair}_cohens_d"
            sig_col  = f"{pair}_sig"

            pair_label = pair.replace("_vs_", " vs ")

            if padj_col in row and pd.notna(row[padj_col]):
                out_row[f"{pair_label} p_adj"] = _format_p(float(row[padj_col]))
                out_row[f"{pair_label} sig"]   = row.get(sig_col, "—")
            else:
                out_row[f"{pair_label} p_adj"] = "—"
                out_row[f"{pair_label} sig"]   = "—"

            if d_col in row and pd.notna(row[d_col]):
                out_row[f"{pair_label} d"] = f"{float(row[d_col]):.2f}"
            else:
                out_row[f"{pair_label} d"] = "—"

        rows.append(out_row)

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"{out_prefix}{suffix}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n  -> ANOVA-Tabelle: {out_path}")
    print(out_df.to_string(index=False).encode("cp1252", errors="replace").decode("cp1252"))


# ── Tabelle 3: Clustering-Metriken (Ward k=3/4/5) ────────────────────────────

def export_metrics(path: Path, out_prefix: str = "apa_tabelle_metriken_stil", selected_k: int | None = None) -> None:
    df = pd.read_csv(path)

    suffix = path.stem.replace("fatigue_cluster_results", "").replace("cluster_results", "")

    # Nur Ward
    ward = df[(df["method"] == "hierarchical") & (df["linkage"] == "ward")].copy()
    if ward.empty:
        print("  WARNUNG: Keine Ward-Ergebnisse in cluster_results gefunden.")
        return

    # Gewählte k: explizit übergeben > is_best aus CSV
    if selected_k is not None:
        best_k = selected_k
    else:
        best_rows = ward[ward["is_best"] == True]
        best_k = int(best_rows["k"].iloc[0]) if not best_rows.empty else None

    # k-Fenster: sel_k-1, sel_k, sel_k+1
    if best_k is not None:
        k_show = [best_k - 1, best_k, best_k + 1]
    else:
        k_show = sorted(ward["k"].unique())[:3]

    ward = ward[ward["k"].isin(k_show)].sort_values("k")

    rows = []
    for _, r in ward.iterrows():
        k_val = int(r["k"])
        is_sel = (best_k is not None and k_val == best_k)
        rows.append({
            "k":              k_val,
            "Silhouette":     f"{r['silhouette']:.3f}",
            "Davies-Bouldin": f"{r['davies_bouldin']:.3f}",
            "Cal.-Harabasz":  f"{r['calinski_harabasz']:.1f}",
            "Gewählt":        "x" if is_sel else "",
        })

    out_df = pd.DataFrame(rows)
    out_path = DATA_DIR / f"{out_prefix}{suffix}.csv"
    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"  -> Metriken-Tabelle (Ward): {out_path}")
    print(out_df.to_string(index=False).encode("cp1252", errors="replace").decode("cp1252"))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=== APA-Tabellen Export ===\n")

    # ── Stil-Clustering ───────────────────────────────────────────────────────
    print("--- Stil-Clustering ---")

    desc_stil = _find_latest("descriptive_stats_clusters*.csv")
    if desc_stil:
        print(f"Lese: {desc_stil.name}")
        export_deskriptiv(desc_stil)
    else:
        print("WARNUNG: Keine descriptive_stats_clusters*.csv gefunden.")

    anova_stil = _find_latest("anova_results_hierarchisch_ward_k4*.csv")
    if anova_stil is None:
        anova_stil = _find_latest("anova_results*.csv")
        # Fatigue-ANOVA ausschliessen
        if anova_stil and "k3" in anova_stil.stem and "fatigue" not in anova_stil.stem:
            anova_stil = None
    if anova_stil:
        print(f"\nLese: {anova_stil.name}")
        export_anova(anova_stil,
                     desc_pattern="descriptive_stats_clusters*.csv",
                     out_prefix="apa_tabelle_anova_stil")
    else:
        print("WARNUNG: Keine Stil-ANOVA-Datei gefunden.")

    # fatigue_cluster_results ausschliessen; suffixierte Datei bevorzugen (hat korrektes is_best)
    metrics_stil_candidates = sorted(
        [f for f in DATA_DIR.glob("cluster_results*.csv") if "fatigue" not in f.name],
        key=lambda f: (len(f.stem), f.stat().st_mtime),  # laengerer Name (mit Suffix) zuerst
        reverse=True,
    )
    metrics_stil = metrics_stil_candidates[0] if metrics_stil_candidates else None
    if metrics_stil:
        print(f"\nLese: {metrics_stil.name}")
        export_metrics(metrics_stil, out_prefix="apa_tabelle_metriken_stil", selected_k=4)
    else:
        print("WARNUNG: Keine cluster_results*.csv gefunden.")

    # ── Fatigue-Clustering ────────────────────────────────────────────────────
    print("\n--- Fatigue-Clustering ---")

    desc_fat = _find_latest("descriptive_stats_fatigue_clusters*.csv")
    if desc_fat:
        print(f"Lese: {desc_fat.name}")
        export_deskriptiv_fatigue(desc_fat)
    else:
        print("WARNUNG: Keine descriptive_stats_fatigue_clusters*.csv gefunden.")

    anova_fat = _find_latest("anova_results_hierarchisch_ward_k3*.csv")
    if anova_fat:
        print(f"\nLese: {anova_fat.name}")
        export_anova(anova_fat,
                     desc_pattern="descriptive_stats_fatigue_clusters*.csv",
                     out_prefix="apa_tabelle_anova_fatigue")
    else:
        print("WARNUNG: Keine Fatigue-ANOVA-Datei (k3) gefunden.")

    metrics_fat = _find_latest("fatigue_cluster_results*.csv")
    if metrics_fat:
        print(f"\nLese: {metrics_fat.name}")
        export_metrics(metrics_fat, out_prefix="apa_tabelle_metriken_fatigue")
    else:
        print("WARNUNG: Keine fatigue_cluster_results*.csv gefunden.")

    print("\n=== Fertig ===")
    print("Tipp: CSV in Google Sheets importieren, Tabelle markieren, in Google Docs einfuegen")


if __name__ == "__main__":
    main()
