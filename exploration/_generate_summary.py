"""Erstellt ergebnisse_zusammenfassung.md aus allen Pipeline-CSVs."""
import pandas as pd
import numpy as np
from pathlib import Path

D   = Path(__file__).resolve().parents[1] / "Outputs" / "Data"
OUT = Path(__file__).resolve().parents[1] / "Outputs" / "ergebnisse_zusammenfassung.md"

def read(f): return pd.read_csv(D / f)

lines = []

def h(t, level=1):
    lines.append("")
    lines.append("#" * level + " " + t)
    lines.append("")

def rule():
    lines.append("---")
    lines.append("")


# ============================================================
# 0. STICHPROBE
# ============================================================
h("STICHPROBE")

df = read("descriptive_stats_sample.csv")
lines.append(f"  {'Variable':<30}  {'N':>5}  {'MW':>8}  {'SD':>8}  {'Min':>8}  {'Max':>8}")
lines.append("  " + "-" * 72)
for _, row in df.iterrows():
    lines.append(
        f"  {str(row['Variable']):<30}  {int(row['N']):>5}  {row['MW']:>8.3f}"
        f"  {row['SD']:>8.3f}  {row['Min']:>8.3f}  {row['Max']:>8.3f}"
    )

rule()

# ============================================================
# 1. SPEED-KORRELATIONSANALYSE
# ============================================================
h("VORANALYSE: SPEED-KORRELATION (km 1.0, N=60)")

df = read("speed_correlation_summary.csv")
lines.append(f"  {'Variable':<12}  {'r':>8}  {'R²':>8}  {'p':>10}  Interpretation")
lines.append("  " + "-" * 55)
for _, row in df.iterrows():
    lines.append(
        f"  {row['Variable']:<12}  {row['r']:>8.3f}  {row['R2']:>8.3f}"
        f"  {row['p_value']:>10.4f}  {row['Interpretation']}"
    )

rule()

# ============================================================
# 2. STIL-CLUSTERING
# ============================================================
h("STIL-CLUSTERING — Ward hierarchisch, k=4, Rohdaten (DF + SF_norm, km 1.0)")

# 2a Cluster-Größen
h("Cluster-Größen", 2)
df_desc = read("descriptive_stats_clusters_hierarchisch_ward_k4_roh.csv")
for _, r in df_desc.iterrows():
    lines.append(f"  {r['Cluster']}: N = {int(r['N'])}")

# 2b Metriken Ward
h("Clustering-Metriken (Ward, alle k)", 2)
cr = read("cluster_results.csv")
ward = cr[(cr["method"] == "hierarchical") & (cr["linkage"] == "ward")].copy()
lines.append(f"  {'k':>4}  {'Silhouette':>12}  {'Davies-Bouldin':>16}  {'Calinski-Harabasz':>18}  best")
lines.append("  " + "-" * 60)
for _, r in ward.iterrows():
    best = "  <- GEWÄHLT" if int(r["k"]) == 4 else ("  <- best" if r["is_best"] else "")
    lines.append(
        f"  {int(r['k']):>4}  {r['silhouette']:>12.3f}  {r['davies_bouldin']:>16.3f}"
        f"  {r['calinski_harabasz']:>18.1f}{best}"
    )

# 2c Silhouette pro Cluster
h("Silhouette pro Cluster (Stil)", 2)
df_sil = read("silhouette_summary_stil.csv")
lines.append(f"  {'Cluster':<15}  {'N':>4}  {'MW':>8}  {'SD':>8}  {'Min':>8}  {'Max':>8}  {'Grenzf.<0.2':>12}")
lines.append("  " + "-" * 70)
for _, r in df_sil.iterrows():
    g_col = [c for c in r.index if "Grenzf" in c]
    g = int(r[g_col[0]]) if g_col else "?"
    lines.append(
        f"  {str(r['Cluster']):<15}  {int(r['N']):>4}  {r['MW']:>8.3f}"
        f"  {r['SD']:>8.3f}  {r['Min']:>8.3f}  {r['Max']:>8.3f}  {g:>12}"
    )

# 2d Deskriptive Statistik
h("Deskriptive Statistik pro Cluster (MW ± SD)", 2)
cluster_ids = list(df_desc["Cluster"])
header = f"  {'Variable':<32}" + "".join([f"  {c:>22}" for c in cluster_ids])
lines.append(header)
lines.append("  " + "-" * (32 + 24 * len(cluster_ids)))

mw_cols = [c for c in df_desc.columns if c.endswith("_MW")]
for col in mw_cols:
    feat   = col.replace("_MW", "")
    sd_col = feat + "_SD"
    if sd_col not in df_desc.columns:
        continue
    label = feat.replace("_", " ")
    row_str = f"  {label:<32}"
    for _, r in df_desc.iterrows():
        row_str += f"  {r[col]:>10.3f} ± {r[sd_col]:<8.3f}"
    lines.append(row_str)

# 2e ANOVA
h("ANOVA-Ergebnisse Stil-Clustering", 2)
df_anova = read("anova_results_hierarchisch_ward_k4_roh.csv")
lines.append(f"  {'Kategorie':<15}  {'Feature':<30}  {'Test':<10}  {'F':>8}  {'p':>8}  {'eta2':>6}  sig")
lines.append("  " + "-" * 85)
for _, r in df_anova.iterrows():
    lines.append(
        f"  {str(r['Kategorie']):<15}  {str(r['Feature']):<30}  {str(r['Test']):<10}"
        f"  {r['F']:>8.3f}  {r['p']:>8.4f}  {r['eta2']:>6.3f}  {r['sig']}"
    )

# 2f Cohen's d
h("Cohen's d (alle Cluster-Paare) — Stil-Clustering", 2)
d_cols = [c for c in df_anova.columns if "cohens_d" in c]
if d_cols:
    pair_labels = [c.replace("_cohens_d", "").replace("Cluster_", "C") for c in d_cols]
    header = f"  {'Feature':<30}" + "".join([f"  {p:>12}" for p in pair_labels])
    lines.append(header)
    lines.append("  " + "-" * (30 + 14 * len(d_cols)))
    for _, r in df_anova.iterrows():
        row_str = f"  {str(r['Feature']):<30}"
        for c in d_cols:
            val = r[c]
            row_str += f"  {val:>12.3f}" if pd.notna(val) else f"  {'—':>12}"
        lines.append(row_str)

# 2g Tukey signifikant
h("Tukey HSD — signifikante Vergleiche (Stil)", 2)
df_tuk = read("tukey_results_hierarchisch_ward_k4_roh.csv")
sig = df_tuk[df_tuk["signifikant"] == True]
if len(sig) == 0:
    lines.append("  (keine signifikanten Vergleiche)")
else:
    lines.append(
        f"  {'Kategorie':<15}  {'Feature':<25}  {'C_A':>5}  {'C_B':>5}"
        f"  {'Diff':>8}  {'p_adj':>8}  {'CI_low':>8}  {'CI_high':>8}"
    )
    lines.append("  " + "-" * 90)
    for _, r in sig.iterrows():
        cat = str(r["Kategorie"]) if "Kategorie" in r.index else ""
        lines.append(
            f"  {cat:<15}  {str(r['Feature']):<25}  {str(r['Cluster_1']):>5}  {str(r['Cluster_2']):>5}"
            f"  {r['Mittelwertdiff']:>8.3f}  {r['p_adj']:>8.4f}"
            f"  {r['CI_low']:>8.3f}  {r['CI_high']:>8.3f}"
        )

# 2h Chi-Quadrat
h("Chi-Quadrat kategoriale Variablen (Stil)", 2)
df_chi = read("chi2_results_hierarchisch_ward_k4_roh.csv")
lines.append(f"  {'Variable':<20}  {'chi2':>8}  {'df':>4}  {'p':>8}  sig")
lines.append("  " + "-" * 48)
for _, r in df_chi.iterrows():
    lines.append(
        f"  {str(r['Variable']):<20}  {r['chi2']:>8.3f}  {int(r['df']):>4}  {r['p']:>8.4f}  {r['sig']}"
    )

rule()

# ============================================================
# 3. FATIGUE-CLUSTERING
# ============================================================
h("FATIGUE-CLUSTERING — Ward hierarchisch, k=3, Fatigue-Features (Delta+Slope DF+SF)")

# 3a Cluster-Größen
h("Cluster-Größen", 2)
df_fdesc = read("descriptive_stats_fatigue_clusters_hierarchisch_ward_k3_roh.csv")
for _, r in df_fdesc.iterrows():
    lines.append(f"  {r['Cluster']}: N = {int(r['N'])}")

# 3b Metriken Ward
h("Clustering-Metriken (Ward, alle k)", 2)
fcr = read("fatigue_cluster_results.csv")
fward = fcr[(fcr["method"] == "hierarchical") & (fcr["linkage"] == "ward")].copy()
lines.append(f"  {'k':>4}  {'Silhouette':>12}  {'Davies-Bouldin':>16}  {'Calinski-Harabasz':>18}  best")
lines.append("  " + "-" * 60)
for _, r in fward.iterrows():
    best = "  <- GEWÄHLT" if int(r["k"]) == 3 else ("  <- best" if r["is_best"] else "")
    lines.append(
        f"  {int(r['k']):>4}  {r['silhouette']:>12.3f}  {r['davies_bouldin']:>16.3f}"
        f"  {r['calinski_harabasz']:>18.1f}{best}"
    )

# 3c Silhouette pro Cluster
h("Silhouette pro Cluster (Fatigue)", 2)
df_fsil = read("silhouette_summary_fatigue.csv")
lines.append(f"  {'Cluster':<15}  {'N':>4}  {'MW':>8}  {'SD':>8}  {'Min':>8}  {'Max':>8}  {'Grenzf.<0.2':>12}")
lines.append("  " + "-" * 70)
for _, r in df_fsil.iterrows():
    g_col = [c for c in r.index if "Grenzf" in c]
    g = int(r[g_col[0]]) if g_col else "?"
    lines.append(
        f"  {str(r['Cluster']):<15}  {int(r['N']):>4}  {r['MW']:>8.3f}"
        f"  {r['SD']:>8.3f}  {r['Min']:>8.3f}  {r['Max']:>8.3f}  {g:>12}"
    )

# 3d Deskriptive Statistik
h("Deskriptive Statistik pro Cluster (MW ± SD)", 2)
cluster_ids_f = list(df_fdesc["Cluster"])
header = f"  {'Variable':<35}" + "".join([f"  {c:>22}" for c in cluster_ids_f])
lines.append(header)
lines.append("  " + "-" * (35 + 24 * len(cluster_ids_f)))

mw_cols_f = [c for c in df_fdesc.columns if c.endswith("_MW")]
for col in mw_cols_f:
    feat   = col.replace("_MW", "")
    sd_col = feat + "_SD"
    if sd_col not in df_fdesc.columns:
        continue
    label = feat.replace("_", " ")
    row_str = f"  {label:<35}"
    for _, r in df_fdesc.iterrows():
        row_str += f"  {r[col]:>10.3f} ± {r[sd_col]:<8.3f}"
    lines.append(row_str)

# 3e ANOVA
h("ANOVA-Ergebnisse Fatigue-Clustering", 2)
df_fanova = read("anova_results_hierarchisch_ward_k3_roh.csv")
lines.append(f"  {'Kategorie':<15}  {'Feature':<30}  {'Test':<10}  {'F':>8}  {'p':>8}  {'eta2':>6}  sig")
lines.append("  " + "-" * 85)
for _, r in df_fanova.iterrows():
    lines.append(
        f"  {str(r['Kategorie']):<15}  {str(r['Feature']):<30}  {str(r['Test']):<10}"
        f"  {r['F']:>8.3f}  {r['p']:>8.4f}  {r['eta2']:>6.3f}  {r['sig']}"
    )

# 3f Cohen's d
h("Cohen's d (alle Cluster-Paare) — Fatigue-Clustering", 2)
fd_cols = [c for c in df_fanova.columns if "cohens_d" in c]
if fd_cols:
    pair_labels = [c.replace("_cohens_d", "").replace("Cluster_", "C") for c in fd_cols]
    header = f"  {'Feature':<30}" + "".join([f"  {p:>12}" for p in pair_labels])
    lines.append(header)
    lines.append("  " + "-" * (30 + 14 * len(fd_cols)))
    for _, r in df_fanova.iterrows():
        row_str = f"  {str(r['Feature']):<30}"
        for c in fd_cols:
            val = r[c]
            row_str += f"  {val:>12.3f}" if pd.notna(val) else f"  {'—':>12}"
        lines.append(row_str)

# 3g Tukey signifikant
h("Tukey HSD — signifikante Vergleiche (Fatigue)", 2)
df_ftuk = read("tukey_results_hierarchisch_ward_k3_roh.csv")
fsig = df_ftuk[df_ftuk["signifikant"] == True]
if len(fsig) == 0:
    lines.append("  (keine signifikanten Vergleiche)")
else:
    lines.append(
        f"  {'Kategorie':<15}  {'Feature':<25}  {'C_A':>5}  {'C_B':>5}"
        f"  {'Diff':>8}  {'p_adj':>8}  {'CI_low':>8}  {'CI_high':>8}"
    )
    lines.append("  " + "-" * 90)
    for _, r in fsig.iterrows():
        cat = str(r["Kategorie"]) if "Kategorie" in r.index else ""
        lines.append(
            f"  {cat:<15}  {str(r['Feature']):<25}  {str(r['Cluster_1']):>5}  {str(r['Cluster_2']):>5}"
            f"  {r['Mittelwertdiff']:>8.3f}  {r['p_adj']:>8.4f}"
            f"  {r['CI_low']:>8.3f}  {r['CI_high']:>8.3f}"
        )

# 3h Chi-Quadrat
h("Chi-Quadrat kategoriale Variablen (Fatigue)", 2)
df_fchi = read("chi2_results_hierarchisch_ward_k3_roh.csv")
lines.append(f"  {'Variable':<20}  {'chi2':>8}  {'df':>4}  {'p':>8}  sig")
lines.append("  " + "-" * 48)
for _, r in df_fchi.iterrows():
    lines.append(
        f"  {str(r['Variable']):<20}  {r['chi2']:>8.3f}  {int(r['df']):>4}  {r['p']:>8.4f}  {r['sig']}"
    )

rule()

# ============================================================
# 4. KREUZTABELLE
# ============================================================
h("KREUZTABELLE: Laufstil-Cluster vs. Fatigue-Cluster")

df_cross = read("chi2_style_vs_fatigue_hierarchisch_ward_k4_roh_vs_hierarchisch_ward_k3_roh.csv")
for _, r in df_cross.iterrows():
    lines.append(f"  chi2={r['chi2']:.3f}, df={int(r['df'])}, p={r['p']:.4f}, sig={r['sig']}")

lines.append("")

# Kreuztabelle aus Labels rekonstruieren
lbl_stil    = pd.read_csv(D / "cluster_labels.csv")
lbl_fatigue = pd.read_csv(D / "fatigue_cluster_labels.csv")

# Spaltennamen normalisieren
for df_lbl, name in [(lbl_stil, "stil"), (lbl_fatigue, "fatigue")]:
    if "cluster_label" not in df_lbl.columns:
        other = [c for c in df_lbl.columns if c != "Subject"]
        df_lbl.rename(columns={other[0]: "cluster_label"}, inplace=True)

merged = lbl_stil.merge(lbl_fatigue, on="Subject", suffixes=("_stil", "_fatigue"))
ct = pd.crosstab(
    merged["cluster_label_stil"],
    merged["cluster_label_fatigue"],
    rownames=["Stil-Cluster"],
    colnames=["Fatigue-Cluster"],
)
lines.append(ct.to_string())

rule()

# Schreiben
out_text = "\n".join(lines)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(out_text, encoding="utf-8")
print(f"Fertig -> {OUT}")
print(f"Zeilen: {len(lines)}")
