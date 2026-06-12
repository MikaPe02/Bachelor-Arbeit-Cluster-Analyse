# src/extension/descriptive.py
#
# ZWECK: Deskriptive Statistik der Cluster nach Methodenwahl.
#        Wird aus main.py aufgerufen, nachdem Labels und Methode feststehen.
#
# EINSTIEGSPUNKT:
#   describe_clusters(df, labels, selection, cfg)
#
# OUTPUT:
#   Terminal-Ausgabe + Outputs/Data/descriptive_stats_clusters.csv

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats as scipy_stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd


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


def _file_suffix(selection: dict) -> str:
    method = selection.get("method", "unbekannt")
    speed_flag = "speedber" if selection.get("speed_corrected") else "roh"
    if method == "kmeans":
        return f"_kmeans_k{selection.get('k', '?')}_{speed_flag}"
    if method == "hierarchical":
        linkage = selection.get("linkage", "")
        return f"_hierarchisch_{linkage}_k{selection.get('k', '?')}_{speed_flag}"
    if method == "hdbscan":
        return f"_hdbscan_mcs{selection.get('min_cluster_size', '?')}_{speed_flag}"
    return f"_{method}_{speed_flag}"


def _method_label(selection: dict) -> str:
    method = selection.get("method", "")
    if method == "kmeans":
        return f"k-Means, k={selection.get('k', '?')}"
    if method == "hierarchical":
        linkage = selection.get("linkage", "").capitalize()
        return f"Hierarchisch ({linkage}), k={selection.get('k', '?')}"
    if method == "hdbscan":
        return f"HDBSCAN, mcs={selection.get('min_cluster_size', '?')}"
    return method


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


# ── Plots ─────────────────────────────────────────────────────────────────────

_PALETTE = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377", "#BBBBBB"]

_FEATURE_LABELS: dict[str, str] = {
    "DF_start":  "DF Start",
    "DF_end":    "DF Ende",
    "Delta_DF":  "Δ DF",
    "Slope_DF":  "Slope DF",
    "SF_start":  "SF_norm Start",
    "SF_end":    "SF_norm Ende",
    "Delta_SF":  "Δ SF_norm",
    "Slope_SF":  "Slope SF_norm",
}


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _plot_fatigue_boxplots(
    df_fatigue: pd.DataFrame,
    fatigue_cols: list[str],
    cluster_ids: list,
    method_lbl: str,
    suffix: str,
    cfg,
) -> None:
    """Speichert jeden Boxplot einzeln in Outputs/Plots/Boxplots/."""
    import matplotlib.lines as mlines

    _BP_RCPARAMS = {
        "font.size": 11, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "legend.fontsize": 10,
    }

    cols_present = [c for c in fatigue_cols if c in df_fatigue.columns]

    cluster_str = [str(c) for c in cluster_ids]
    colors = [_PALETTE[i % len(_PALETTE)] for i in range(len(cluster_ids))]

    # Unterordner
    box_dir = cfg.OUTPUT_PLOTS_DIR / "Boxplots"
    box_dir.mkdir(parents=True, exist_ok=True)

    for col in cols_present:
        plt.rcParams.update(_BP_RCPARAMS)

        data_per_cluster = [
            df_fatigue.loc[df_fatigue["cluster_label"] == cid, col].dropna().values
            for cid in cluster_ids
        ]

        fig, ax = plt.subplots(figsize=(16 / 2.54, 12 / 2.54))

        bp = ax.boxplot(
            data_per_cluster,
            patch_artist=True,
            widths=0.5,
            medianprops=dict(color="black", linewidth=1.5),
            whiskerprops=dict(linewidth=1.0),
            capprops=dict(linewidth=1.0),
            flierprops=dict(marker="o", markersize=4, linestyle="none",
                            markeredgecolor="gray", markerfacecolor="none"),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(cluster_ids) + 1))
        ax.set_xticklabels([str(c) for c in cluster_str], fontsize=10)
        ax.set_ylabel(_FEATURE_LABELS.get(col, col))
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
        ax.grid(True, linewidth=0.5, alpha=0.5)


        fig.tight_layout()
        fname = col.replace(" ", "_").replace("/", "_").replace("Δ", "Delta")
        out = box_dir / f"boxplot_{fname}{suffix}.png"
        fig.savefig(out, dpi=cfg.PLOT_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  -> Boxplot gespeichert: {out}")


# ── Inferenzstatistik ────────────────────────────────────────────────────────

def _run_anova_block(
    df_data: pd.DataFrame,
    cols: list[str],
    cluster_ids: list,
    method_lbl: str,
    title: str,
    category: str,
) -> tuple[list[dict], list[dict]]:
    """
    Fuehrt ANOVA + Tukey fuer alle cols durch.
    Gibt (anova_records, tukey_records) zurueck.
    """
    col_w = {"Feature": 24, "F": 9, "p": 9, "eta2": 7, "sig": 5}
    header = (
        f"  {'Feature':<{col_w['Feature']}}"
        f"{'F':>{col_w['F']}}"
        f"{'p':>{col_w['p']}}"
        f"{'η²':>{col_w['eta2']}}"
        f"{'sig':>{col_w['sig']}}"
    )
    sep = "  " + "-" * sum(col_w.values())
    print(f"\n{title}")
    print(f"  Verfahren : {method_lbl}")
    print(f"\n{sep}")
    print(header)
    print(sep)

    anova_records: list[dict] = []
    tukey_records: list[dict] = []

    for col in cols:
        if col not in df_data.columns:
            continue

        groups = [
            df_data.loc[df_data["cluster_label"] == cid, col].dropna().values
            for cid in cluster_ids
        ]
        valid = [g for g in groups if len(g) >= 2]
        if len(valid) < 2:
            continue

        f_val, p_val = scipy_stats.f_oneway(*valid)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

        # η² = SS_between / SS_total
        all_vals  = df_data.loc[df_data["cluster_label"].isin(cluster_ids), col].dropna()
        grand_mean = all_vals.mean()
        ss_total   = ((all_vals - grand_mean) ** 2).sum()
        ss_between = sum(
            len(g) * (g.mean() - grand_mean) ** 2
            for g in valid
        )
        eta2 = round(ss_between / ss_total, 4) if ss_total > 0 else float("nan")

        print(
            f"  {col:<{col_w['Feature']}}"
            f"{f_val:>{col_w['F']}.3f}"
            f"{p_val:>{col_w['p']}.4f}"
            f"{eta2:>{col_w['eta2']}.3f}"
            f"  {sig}"
        )

        anova_records.append({
            "Kategorie": category,
            "Verfahren": method_lbl,
            "Feature": col,
            "F": round(f_val, 4),
            "p": round(p_val, 4),
            "eta2": eta2,
            "sig": sig,
        })

        if p_val < 0.05:
            vals       = df_data[col].dropna()
            grp_labels = df_data.loc[vals.index, "cluster_label"].astype(str)
            tukey      = pairwise_tukeyhsd(vals.values, grp_labels.values, alpha=0.05)

            t_col_w = {"pair": 28, "diff": 12, "p_adj": 10, "ci": 26, "sig": 6}
            t_sep   = "    " + "-" * sum(t_col_w.values())
            t_header = (
                f"    {'Vergleich':<{t_col_w['pair']}}"
                f"{'Diff':>{t_col_w['diff']}}"
                f"{'p_adj':>{t_col_w['p_adj']}}"
                f"  {'95%-KI':<{t_col_w['ci']}}"
                f"{'sig':>{t_col_w['sig']}}"
            )
            print(t_sep)
            print(t_header)
            print(t_sep)

            for row in tukey.summary().data[1:]:
                g1, g2, meandiff, p_adj, lower, upper, reject = row
                pair   = f"{g1} vs {g2}"
                ci_str = f"[{float(lower):+.4f}, {float(upper):+.4f}]"
                sig_t  = "ja" if reject else "nein"
                print(
                    f"    {pair:<{t_col_w['pair']}}"
                    f"{float(meandiff):>{t_col_w['diff']:d}.5f}"
                    f"{float(p_adj):>{t_col_w['p_adj']}.4f}"
                    f"  {ci_str:<{t_col_w['ci']}}"
                    f"  {sig_t}"
                )
                tukey_records.append({
                    "Kategorie": category,
                    "Verfahren": method_lbl,
                    "Feature": col,
                    "Cluster_1": str(g1),
                    "Cluster_2": str(g2),
                    "Mittelwertdiff": round(float(meandiff), 5),
                    "p_adj": round(float(p_adj), 4),
                    "CI_low": round(float(lower), 5),
                    "CI_high": round(float(upper), 5),
                    "signifikant": bool(reject),
                })
            print(t_sep)

    print(sep)
    print("  Signifikanzniveaus: * p<.05  ** p<.01  *** p<.001  n.s. = nicht signifikant")

    return anova_records, tukey_records


def _test_categorical_between_clusters(
    df: pd.DataFrame,
    cat_cols: list[str],
    cluster_ids: list,
    method_lbl: str,
    suffix: str,
    cfg,
) -> None:
    """Chi-Quadrat-Test fuer kategoriale Variablen zwischen Clustern."""
    available = [c for c in cat_cols if c in df.columns]
    if not available:
        return

    print(f"\n=== Chi-Quadrat-Test: Kategoriale Variablen ===")
    print(f"  Verfahren : {method_lbl}")

    col_w = {"Variable": 24, "chi2": 9, "p": 9, "df": 5, "sig": 5}
    header = (
        f"  {'Variable':<{col_w['Variable']}}"
        f"{'χ²':>{col_w['chi2']}}"
        f"{'p':>{col_w['p']}}"
        f"{'df':>{col_w['df']}}"
        f"{'sig':>{col_w['sig']}}"
    )
    sep = "  " + "-" * sum(col_w.values())
    print(f"\n{sep}")
    print(header)
    print(sep)

    records: list[dict] = []

    for col in available:
        sub = df[["cluster_label", col]].dropna()
        contingency = pd.crosstab(sub["cluster_label"], sub[col])

        # mind. 2 Zeilen und 2 Spalten
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            continue

        chi2, p_val, dof, _ = scipy_stats.chi2_contingency(contingency)
        sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

        print(
            f"  {col:<{col_w['Variable']}}"
            f"{chi2:>{col_w['chi2']}.3f}"
            f"{p_val:>{col_w['p']}.4f}"
            f"{dof:>{col_w['df']}}"
            f"  {sig}"
        )

        # Kontingenztabelle im Terminal
        print(f"\n  Kontingenztabelle ({col}):")
        print("  " + contingency.to_string().replace("\n", "\n  "))
        print()

        records.append({
            "Verfahren": method_lbl,
            "Variable": col,
            "chi2": round(chi2, 4),
            "p": round(p_val, 4),
            "df": dof,
            "sig": sig,
        })

    print(sep)
    print("  Signifikanzniveaus: * p<.05  ** p<.01  *** p<.001  n.s. = nicht signifikant")

    if records:
        cfg.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out = cfg.OUTPUT_DATA_DIR / f"chi2_results{suffix}.csv"
        pd.DataFrame(records).to_csv(out, index=False)
        print(f"\n  -> Chi-Quadrat-Ergebnisse gespeichert: {out}")


def _test_fatigue_between_clusters(
    df_fatigue: pd.DataFrame,
    fatigue_cols: list[str],
    cluster_ids: list,
    method_lbl: str,
    suffix: str,
    cfg,
    df_biomech: pd.DataFrame | None = None,
    biomech_cols: list[str] | None = None,
    boxplot_cols: list[str] | None = None,
) -> None:
    """
    ANOVA + Tukey fuer Fatigue-Features und optional biomechanische Parameter.
    Alle Ergebnisse in einer CSV.
    """
    print("\n=== Inferenzstatistik: ANOVA + Tukey HSD ===")
    # Block 1: Biomechanische Parameter
    all_anova: list[dict] = []
    all_tukey: list[dict] = []

    if df_biomech is not None and biomech_cols:
        a, t = _run_anova_block(
            df_biomech, biomech_cols, cluster_ids, method_lbl,
            "=== ANOVA: Biomechanische Parameter & Anthropometrie ===", "Biomechanik"
        )
        all_anova.extend(a)
        all_tukey.extend(t)

    # Block 2: Fatigue-Features
    a, t = _run_anova_block(
        df_fatigue, fatigue_cols, cluster_ids, method_lbl,
        "=== ANOVA: Fatigue-Features ===", "Fatigue"
    )
    all_anova.extend(a)
    all_tukey.extend(t)

    # Tukey-Ergebnisse als extra Spalten in anova_records einbauen
    if all_tukey:
        df_tukey = pd.DataFrame(all_tukey)
        for _, trow in df_tukey.iterrows():
            pair_key = f"{trow['Cluster_1']}_vs_{trow['Cluster_2']}".replace(" ", "_")
            for rec in all_anova:
                if rec["Feature"] == trow["Feature"] and rec["Kategorie"] == trow["Kategorie"]:
                    rec[f"{pair_key}_p_adj"] = trow["p_adj"]
                    rec[f"{pair_key}_sig"]   = "ja" if trow["signifikant"] else "nein"

    # Speichern
    cfg.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_anova = cfg.OUTPUT_DATA_DIR / f"anova_results{suffix}.csv"
    pd.DataFrame(all_anova).to_csv(out_anova, index=False)
    print(f"\n  -> ANOVA-Ergebnisse gespeichert: {out_anova}")

    if all_tukey:
        out_tukey = cfg.OUTPUT_DATA_DIR / f"tukey_results{suffix}.csv"
        pd.DataFrame(all_tukey).to_csv(out_tukey, index=False)
        print(f"  -> Tukey-Ergebnisse gespeichert: {out_tukey}")
    else:
        print("  -> Kein signifikantes ANOVA-Ergebnis: kein Tukey post-hoc berechnet.")

    # Boxplot nur fuer Fatigue-Features
    fatigue_tukey = [r for r in all_tukey if r["Kategorie"] == "Fatigue"]
    plot_cols = boxplot_cols if boxplot_cols is not None else fatigue_cols
    _plot_fatigue_boxplots(df_fatigue, plot_cols, cluster_ids, method_lbl, suffix, cfg)


# ── Hauptfunktion ─────────────────────────────────────────────────────────────

def describe_clusters(
    df: pd.DataFrame,
    labels: pd.Series,
    selection: dict,
    cfg,
) -> None:
    """
    Deskriptive Statistik pro Cluster nach Methodenwahl.

    Parameters
    ----------
    df        : langer Datensatz (Subject x km), muss DF, SF_norm, speed_ms enthalten
    labels    : pd.Series mit Index=Subject, Values=Cluster-Label
    selection : dict mit 'method', 'linkage', 'k' (Ausgabe von select_clustering)
    cfg       : config-Modul (benoetigt SUBJECTS_CSV, OUTPUT_DATA_DIR)
    """
    print("\n=== Schritt 5e: Deskriptive Statistik pro Cluster ===")

    method_lbl = _method_label(selection)

    # km 1.0 snapshot
    km1 = df[np.abs(df["km"] - 1.0) <= 1e-6].copy()

    # Cluster-Labels mergen
    df_labels = labels.reset_index()
    df_labels.columns = ["Subject", "cluster_label"]
    km1 = km1.merge(df_labels, on="Subject", how="left")

    # Probanden-Metadaten mergen (Körpergröße, Gewicht, Beinlänge, Speed)
    meta_cols = ["Subject", "body_height_m", "weight_kg", "leg_length_m", "speed_ms", "SF_hz_km1"]
    if cfg.SUBJECTS_CSV.exists():
        subjects = pd.read_csv(cfg.SUBJECTS_CSV)
        available = [c for c in meta_cols if c in subjects.columns]
        km1 = km1.merge(subjects[available], on="Subject", how="left")

    cluster_ids = sorted(km1["cluster_label"].dropna().unique(), key=str)

    # Fatigue-Features laden und mit Cluster-Labels mergen
    fatigue_cols = [
        "DF_start", "DF_end", "Delta_DF", "Slope_DF",
        "SF_start", "SF_end", "Delta_SF", "Slope_SF",
    ]
    # Nur Ermüdungsparameter für Boxplots (Start/Ende sind Clustering-Variablen)
    boxplot_cols = ["Delta_DF", "Slope_DF", "Delta_SF", "Slope_SF"]
    df_fatigue: pd.DataFrame | None = None
    if hasattr(cfg, "FATIGUE_FEATURES_CSV") and cfg.FATIGUE_FEATURES_CSV.exists():
        df_fatigue = pd.read_csv(cfg.FATIGUE_FEATURES_CSV)
        df_fatigue = df_fatigue.merge(df_labels, on="Subject", how="left")

    print(f"\n  Verfahren : {method_lbl}")
    print(f"  Cluster   : {cluster_ids}")

    all_records: list[dict] = []

    for cid in cluster_ids:
        sub = km1[km1["cluster_label"] == cid]
        n   = len(sub)

        vars_: list[tuple[str, pd.Series]] = [
            ("Duty Factor (DF)", sub["DF"]),
            ("SF_norm",          sub["SF_norm"]),
        ]
        if "speed_ms" in sub.columns:
            vars_.append(("Speed [m/s]", sub["speed_ms"]))
        if "body_height_m" in sub.columns:
            vars_.append(("Körpergröße [m]", sub["body_height_m"]))
        if "weight_kg" in sub.columns:
            vars_.append(("Gewicht [kg]", sub["weight_kg"]))
        if "leg_length_m" in sub.columns:
            vars_.append(("Beinlänge [m]", sub["leg_length_m"]))
        if "SF_hz_km1" in sub.columns:
            vars_.append(("SF bei km 1.0 [Hz]", sub["SF_hz_km1"]))

        # Fatigue-Features an vars_ anhängen (gleiche Tabelle)
        if df_fatigue is not None:
            sub_f = df_fatigue[df_fatigue["cluster_label"] == cid]
            for col in fatigue_cols:
                if col in sub_f.columns:
                    vars_.append((col, sub_f[col]))

        rows = [(lbl, _stats_row(s)) for lbl, s in vars_]
        _print_table(f"  Cluster {cid}  (n = {n}):", rows)

        # Breites Format: eine Zeile pro Cluster, Spalten = Variable_Kennwert
        row_wide: dict = {"Verfahren": method_lbl, "Cluster": str(cid), "N": n}
        for lbl, stats in rows:
            prefix = lbl.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "")
            for kennwert in ("MW", "SD", "Min", "Max"):
                row_wide[f"{prefix}_{kennwert}"] = stats[kennwert]

        all_records.append(row_wide)

    # Dateiname mit Methoden-Suffix
    cfg.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    suffix = _file_suffix(selection)
    out = cfg.OUTPUT_DATA_DIR / f"descriptive_stats_clusters{suffix}.csv"
    pd.DataFrame(all_records).to_csv(out, index=False)
    print(f"\n  -> Gespeichert: {out}")

    # Inferenzstatistik: ANOVA + Tukey
    if df_fatigue is not None:
        biomech_cols = ["DF", "SF_norm"]
        for col in ("speed_ms", "body_height_m", "weight_kg", "leg_length_m"):
            if col in km1.columns:
                biomech_cols.append(col)

        _test_fatigue_between_clusters(
            df_fatigue, fatigue_cols, cluster_ids, method_lbl, suffix, cfg,
            df_biomech=km1, biomech_cols=biomech_cols,
            boxplot_cols=boxplot_cols,
        )

    # Chi-Quadrat: kategoriale Variablen
    cat_cols = ["sex", "dominant_leg", "age"]
    if cfg.SUBJECTS_CSV.exists():
        subjects_cat = pd.read_csv(cfg.SUBJECTS_CSV)
        available_cat = [c for c in cat_cols if c in subjects_cat.columns]
        if available_cat:
            df_cat = km1[["Subject", "cluster_label"]].drop_duplicates()
            df_cat = df_cat.merge(subjects_cat[["Subject"] + available_cat], on="Subject", how="left")
            _test_categorical_between_clusters(df_cat, available_cat, cluster_ids, method_lbl, suffix, cfg)


def compare_style_and_fatigue_clusters(
    style_labels: "pd.Series",
    fatigue_labels: "pd.Series",
    style_selection: dict,
    fatigue_selection: dict,
    cfg,
) -> None:
    """
    Kreuztabelle + Chi-Quadrat-Test: Laufstil-Cluster vs. Fatigue-Cluster.

    Parameters
    ----------
    style_labels   : pd.Series (Index=Subject, Values=Laufstil-Cluster-Label)
    fatigue_labels : pd.Series (Index=Subject, Values=Fatigue-Cluster-Label)
    style_selection  : selection-dict des Laufstil-Clusterings
    fatigue_selection: selection-dict des Fatigue-Clusterings
    cfg            : config-Modul
    """
    print("\n=== Kreuztabelle: Laufstil-Cluster vs. Fatigue-Cluster ===")

    style_lbl   = _method_label(style_selection)
    fatigue_lbl = _method_label(fatigue_selection)
    print(f"  Laufstil-Clustering : {style_lbl}")
    print(f"  Fatigue-Clustering  : {fatigue_lbl}")

    # Beide Labels auf gemeinsame Probanden joinen
    df_s = style_labels.rename("Laufstil").reset_index()
    df_s.columns = ["Subject", "Laufstil"]
    df_f = fatigue_labels.rename("Fatigue").reset_index()
    df_f.columns = ["Subject", "Fatigue"]

    df_merged = df_s.merge(df_f, on="Subject", how="inner")
    n_common = len(df_merged)
    print(f"  Gemeinsame Probanden: {n_common}")

    if n_common < 4:
        print("  -> Zu wenige gemeinsame Probanden fuer Kreuztabelle.")
        return

    df_merged["Laufstil"] = df_merged["Laufstil"].astype(str)
    df_merged["Fatigue"]  = df_merged["Fatigue"].astype(str)

    # Kreuztabelle
    ct = pd.crosstab(
        df_merged["Laufstil"],
        df_merged["Fatigue"],
        rownames=["Laufstil-Cluster"],
        colnames=["Fatigue-Cluster"],
    )

    print("\n  Kreuztabelle (Anzahl Probanden):")
    print("  " + ct.to_string().replace("\n", "\n  "))

    # Zeilenprozente
    ct_pct = ct.div(ct.sum(axis=1), axis=0).round(3) * 100
    print("\n  Zeilenprozente (% pro Laufstil-Cluster):")
    print("  " + ct_pct.to_string().replace("\n", "\n  "))

    # Chi-Quadrat-Test
    chi2, p_val, dof, expected = scipy_stats.chi2_contingency(ct)
    sig = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))

    print(f"\n  Chi-Quadrat-Test:")
    print(f"    χ² = {chi2:.3f},  df = {dof},  p = {p_val:.4f}  {sig}")

    if p_val < 0.05:
        print("  -> Laufstil- und Fatigue-Cluster sind NICHT unabhaengig:")
        print("     Laufstilgruppen unterscheiden sich in ihrer Ermüdungstyp-Verteilung.")
    else:
        print("  -> Laufstil- und Fatigue-Cluster sind UNABHAENGIG:")
        print("     Laufstil erklaert den Ermüdungstyp nicht.")

    # Speichern
    cfg.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    style_suffix   = _file_suffix(style_selection)
    fatigue_suffix = _file_suffix(fatigue_selection)

    out_ct = cfg.OUTPUT_DATA_DIR / f"crosstab_style_vs_fatigue{style_suffix}_vs{fatigue_suffix}.csv"
    ct.to_csv(out_ct)
    print(f"\n  -> Kreuztabelle gespeichert: {out_ct}")

    out_pct = cfg.OUTPUT_DATA_DIR / f"crosstab_pct_style_vs_fatigue{style_suffix}_vs{fatigue_suffix}.csv"
    ct_pct.to_csv(out_pct)
    print(f"  -> Zeilenprozente gespeichert: {out_pct}")

    chi2_rec = [{
        "Laufstil_Verfahren": style_lbl,
        "Fatigue_Verfahren":  fatigue_lbl,
        "chi2": round(chi2, 4),
        "p":    round(p_val, 4),
        "df":   dof,
        "sig":  sig,
    }]
    out_chi2 = cfg.OUTPUT_DATA_DIR / f"chi2_style_vs_fatigue{style_suffix}_vs{fatigue_suffix}.csv"
    pd.DataFrame(chi2_rec).to_csv(out_chi2, index=False)
    print(f"  -> Chi-Quadrat-Ergebnis gespeichert: {out_chi2}")
