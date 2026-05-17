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

        rows = [(lbl, _stats_row(s)) for lbl, s in vars_]
        _print_table(f"  Cluster {cid}  (n = {n}):", rows)

        # Breites Format: eine Zeile pro Cluster, Spalten = Variable_Kennwert
        row_wide: dict = {"Verfahren": method_lbl, "Cluster": str(cid), "N": n}
        for lbl, stats in rows:
            # Spaltenprefix: Sonderzeichen entfernen fuer saubere CSV-Header
            prefix = lbl.replace(" ", "_").replace("[", "").replace("]", "").replace("/", "")
            for kennwert in ("MW", "SD", "Min", "Max"):
                row_wide[f"{prefix}_{kennwert}"] = stats[kennwert]
        all_records.append(row_wide)

    # Speichern
    cfg.OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.OUTPUT_DATA_DIR / "descriptive_stats_clusters.csv"
    pd.DataFrame(all_records).to_csv(out, index=False)
    print(f"\n  -> Gespeichert: {out}")
