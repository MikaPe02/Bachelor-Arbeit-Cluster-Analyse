# src/fatigue/clustering_ui.py
#
# ZWECK: Interaktive Terminal-Funktionen fuer die Clustering-Konfiguration.
#
# FUNKTIONEN:
#   show_metrics_summary(df_results) -> None
#       Gibt beste Konfiguration pro Methode/Linkage tabellarisch aus.
#       Markiert die global beste Konfiguration mit →.
#
#   select_clustering(df_results) -> dict
#       Interaktives Menue: Empfehlung uebernehmen oder selbst waehlen.
#       Gibt {"method": ..., "linkage": ..., "k": ...} zurueck.

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# config aus dem Projektstamm importieren
sys.path.append(str(Path(__file__).resolve().parents[3]))
import config


# ── Hilfsfunktion ─────────────────────────────────────────────────────────────

def _global_best_row(df_results: pd.DataFrame) -> pd.Series:
    """Gibt die global beste Konfiguration zurueck.

    Kriterium: niedrigstes rank_mean ueber alle is_best-Zeilen.
    Fallback auf hoechsten Silhouette-Score falls rank_mean fehlt.
    """
    best_rows = df_results[df_results["is_best"] == True]
    if best_rows.empty:
        best_rows = df_results

    if "rank_mean" in best_rows.columns:
        return best_rows.loc[best_rows["rank_mean"].idxmin()]
    return best_rows.loc[best_rows["silhouette"].idxmax()]


# ── Oeffentliche Funktionen ───────────────────────────────────────────────────

def show_metrics_summary(df_results: pd.DataFrame) -> None:
    """Beste Konfiguration pro Methode/Linkage im Terminal ausgeben.

    Markiert die global beste Konfiguration (niedrigstes rank_mean) mit →.
    """
    print("\n=== Schritt 4c: Metriken-Zusammenfassung ===")

    best_rows   = df_results[df_results["is_best"] == True].copy()
    global_best = _global_best_row(df_results)

    gb_key = (global_best["method"], str(global_best.get("linkage", "")), int(global_best["k"]))

    header = (
        f"  {'':2s}  {'Methode':<14s}  {'Linkage':<10s}  {'k':>2s}"
        f"  {'Silhouette':>10s}  {'Davies-Bouldin':>14s}  {'Calinski-Harabasz':>17s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for _, row in best_rows.iterrows():
        key         = (row["method"], str(row.get("linkage", "")), int(row["k"]))
        marker      = "→" if key == gb_key else " "
        linkage_str = str(row["linkage"]) if pd.notna(row.get("linkage")) else ""
        print(
            f"  {marker}   {row['method']:<14s}  {linkage_str:<10s}  {int(row['k']):>2d}"
            f"  {row['silhouette']:>10.4f}  {row['davies_bouldin']:>14.4f}"
            f"  {row['calinski_harabasz']:>17.4f}"
        )

    print(sep)
    gb_linkage = str(global_best["linkage"]) if pd.notna(global_best.get("linkage")) else "–"
    criterion  = "rank_mean" if "rank_mean" in df_results.columns else "Silhouette"
    print(
        f"  → Global beste Konfiguration ({criterion}): "
        f"{global_best['method']}, Linkage={gb_linkage}, k={int(global_best['k'])}"
    )


def select_clustering(df_results: pd.DataFrame) -> dict:
    """Interaktives Menue zur Wahl der finalen Clustering-Konfiguration.

    Returns
    -------
    dict mit Schluesseln "method", "linkage", "k"
    """
    print("\n=== Schritt 4d: Clustering-Konfiguration waehlen ===")

    global_best = _global_best_row(df_results)
    gb_method   = global_best["method"]
    gb_linkage  = global_best.get("linkage")
    gb_linkage  = str(gb_linkage) if pd.notna(gb_linkage) else None
    gb_k        = int(global_best["k"])

    linkage_label = f", Linkage={gb_linkage}" if gb_linkage else ""
    print(f"  Empfehlung: {gb_method}{linkage_label}, k={gb_k}")

    # ── Hauptauswahl ──────────────────────────────────────────────────────────
    while True:
        print("\n  [1] Empfehlung uebernehmen")
        print("  [2] Selbst waehlen")
        choice = input("  Auswahl: ").strip()
        if choice in ("1", "2"):
            break
        print("  Ungueltige Eingabe, bitte nochmal:")

    if choice == "1":
        return {"method": gb_method, "linkage": gb_linkage, "k": gb_k}

    # ── Methode waehlen ───────────────────────────────────────────────────────
    method_map = {
        "1": ("kmeans",       None),
        "2": ("hierarchical", "ward"),
        "3": ("hierarchical", "complete"),
        "4": ("hierarchical", "average"),
        "5": ("hierarchical", "single"),
    }
    while True:
        print("\n  Methode:")
        print("    [1] k-Means")
        print("    [2] Ward")
        print("    [3] Complete")
        print("    [4] Average")
        print("    [5] Single")
        m_choice = input("  Auswahl: ").strip()
        if m_choice in method_map:
            sel_method, sel_linkage = method_map[m_choice]
            break
        print("  Ungueltige Eingabe, bitte nochmal:")

    # Empfohlenes k fuer die gewaehlte Methode/Linkage ermitteln
    mask = df_results["method"] == sel_method
    if sel_linkage:
        mask &= df_results["linkage"] == sel_linkage
    group_rows = df_results[mask & (df_results["is_best"] == True)]
    if group_rows.empty:
        group_rows = df_results[mask]

    if "rank_mean" in group_rows.columns:
        rec_k = int(group_rows.loc[group_rows["rank_mean"].idxmin(), "k"])
    else:
        rec_k = int(group_rows.loc[group_rows["silhouette"].idxmax(), "k"])

    # ── k waehlen ─────────────────────────────────────────────────────────────
    valid_ks = [int(k) for k in config.K_RANGE]
    while True:
        print(f"\n  k waehlen (empfohlen: k={rec_k}):")
        print(f"    [1] Empfohlenes k uebernehmen (k={rec_k})")
        print(f"    [2] k manuell eingeben (gueltig: {valid_ks})")
        k_choice = input("  Auswahl: ").strip()
        if k_choice == "1":
            sel_k = rec_k
            break
        elif k_choice == "2":
            while True:
                raw = input(f"  k eingeben {valid_ks}: ").strip()
                try:
                    sel_k = int(raw)
                    if sel_k in valid_ks:
                        break
                    print(f"  Ungueltige Eingabe: k={sel_k} nicht in {valid_ks}. Bitte nochmal:")
                except ValueError:
                    print("  Ungueltige Eingabe, bitte nochmal:")
            break
        else:
            print("  Ungueltige Eingabe, bitte nochmal:")

    linkage_label = f", Linkage={sel_linkage}" if sel_linkage else ""
    print(f"\n  Gewaehlt: {sel_method}{linkage_label}, k={sel_k}")
    return {"method": sel_method, "linkage": sel_linkage, "k": sel_k}
