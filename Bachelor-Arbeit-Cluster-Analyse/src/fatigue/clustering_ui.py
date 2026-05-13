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


# ── Hilfsfunktionen ───────────────────────────────────────────────────────────

def _show_plot(path) -> None:
    """Oeffnet einen gespeicherten Plot im Standard-Bildbetrachter."""
    import subprocess, os
    try:
        os.startfile(str(path))
    except AttributeError:
        subprocess.run(["xdg-open", str(path)])


def _show_dendrogram_preview(df_features_z: "pd.DataFrame", linkage_method: str) -> None:
    """Zeigt Dendrogramm ohne Schnittlinie zur Orientierung."""
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import linkage, dendrogram

    X        = df_features_z.drop(columns=["Subject"]).to_numpy(dtype=float)
    subjects = df_features_z["Subject"].tolist()

    Z = linkage(X, method=linkage_method)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(f"Dendrogramm ({linkage_method}) – Orientierung zur k-Wahl")
    dendrogram(Z, labels=subjects, ax=ax, color_threshold=0,
               above_threshold_color="#4477AA")
    ax.set_ylabel("Distanz")
    ax.tick_params(axis="x", labelsize=8, rotation=45)
    plt.setp(ax.get_xticklabels(), ha="right")
    ax.grid(False)
    fig.tight_layout()
    plt.show(block=True)
    plt.close(fig)


def _global_best_row(df_results: pd.DataFrame, verbose: bool = False) -> pd.Series:
    """Gibt die global beste Konfiguration zurueck.

    Kriterium: frische Rang-Aggregation ueber die absoluten Metrikwerte
    (Silhouette hoch, Davies-Bouldin niedrig, Calinski-Harabasz hoch)
    aller is_best-Zeilen (eine beste pro Methode/Linkage-Gruppe).
    So sind rank_mean-Werte verschiedener Gruppen nicht mehr noetig –
    es zaehlen die echten Messwerte im direkten Vergleich.
    Tiebreaker: hoehere Silhouette.

    Parameters
    ----------
    verbose : bool
        True = cross_rank_mean aller is_best-Zeilen ausgeben (Diagnose).
    """
    best_rows = df_results[df_results["is_best"] == True].copy()
    if best_rows.empty:
        best_rows = df_results.copy()

    # Nur Zeilen mit gueltigen Metriken fuer den Vergleich verwenden
    valid = best_rows[best_rows["silhouette"].notna()].copy()
    if valid.empty:
        return best_rows.loc[best_rows["silhouette"].idxmax()]

    # Frische methodenuebergreifende Rang-Aggregation auf absoluten Werten
    valid["_rank_sil"] = valid["silhouette"].rank(ascending=False)
    valid["_rank_db"]  = valid["davies_bouldin"].rank(ascending=True)
    valid["_rank_ch"]  = valid["calinski_harabasz"].rank(ascending=False)
    valid["_cross_rank_mean"] = valid[["_rank_sil", "_rank_db", "_rank_ch"]].mean(axis=1)

    if verbose:
        print("  [DEBUG] _global_best_row() cross_rank_mean:")
        for _, row in valid.iterrows():
            print(f"    {row['method']}, linkage={row.get('linkage')}, "
                  f"sil={row['silhouette']:.4f}, cross_rank={row['_cross_rank_mean']:.3f}")

    return valid.sort_values(
        ["_cross_rank_mean", "silhouette"], ascending=[True, False]
    ).iloc[0]


# ── Oeffentliche Funktionen ───────────────────────────────────────────────────

def show_metrics_summary(df_results: pd.DataFrame) -> None:
    """Beste Konfiguration pro Methode/Linkage im Terminal ausgeben.

    Markiert die global beste Konfiguration (niedrigstes rank_mean) mit →.
    """
    print("\n=== Schritt 4c: Metriken-Zusammenfassung ===")

    best_rows   = df_results[df_results["is_best"] == True].copy()
    global_best = _global_best_row(df_results)

    # HDBSCAN: nur die eine beste Parameterkombination (niedrigstes rank_mean) anzeigen
    hdb = best_rows[best_rows["method"] == "hdbscan"]
    if not hdb.empty and "rank_mean" in hdb.columns and hdb["rank_mean"].notna().any():
        hdb = hdb.loc[[hdb["rank_mean"].idxmin()]]
    best_rows = pd.concat(
        [best_rows[best_rows["method"] != "hdbscan"], hdb]
    ).sort_index()

    def _k_display(row) -> int:
        """k fuer Anzeige: bei HDBSCAN n_clusters_found, sonst k."""
        return int(row["k"]) if pd.notna(row.get("k")) else int(row["n_clusters_found"])

    gb_key = (global_best["method"], str(global_best.get("linkage", "")), _k_display(global_best))

    header = (
        f"  {'':2s}  {'Methode':<14s}  {'Linkage':<10s}  {'k':>2s}"
        f"  {'Silhouette':>10s}  {'Davies-Bouldin':>14s}  {'Calinski-Harabasz':>17s}"
    )
    sep = "  " + "-" * (len(header) - 2)
    print(header)
    print(sep)

    for _, row in best_rows.iterrows():
        key         = (row["method"], str(row.get("linkage", "")), _k_display(row))
        marker      = ">" if key == gb_key else " "
        linkage_str = str(row["linkage"]) if pd.notna(row.get("linkage")) else ""
        print(
            f"  {marker}   {row['method']:<14s}  {linkage_str:<10s}  {_k_display(row):>2d}"
            f"  {row['silhouette']:>10.4f}  {row['davies_bouldin']:>14.4f}"
            f"  {row['calinski_harabasz']:>17.4f}"
        )

    print(sep)
    gb_linkage = str(global_best["linkage"]) if pd.notna(global_best.get("linkage")) else "–"
    criterion  = "rank_mean" if "rank_mean" in df_results.columns else "Silhouette"
    print(
        f"  > Global beste Konfiguration ({criterion}): "
        f"{global_best['method']}, Linkage={gb_linkage}, k={_k_display(global_best)}"
    )


def select_clustering(df_results: pd.DataFrame, df_features_z: pd.DataFrame | None = None, cfg=None) -> dict:
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
    gb_k        = int(global_best["k"]) if pd.notna(global_best.get("k")) else int(global_best["n_clusters_found"])

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
        if gb_method == "hdbscan":
            gb_mcs = int(global_best["min_cluster_size"])
            gb_ms  = None if pd.isna(global_best.get("min_samples")) else int(global_best["min_samples"])
            return {"method": "hdbscan", "linkage": None, "k": gb_k,
                    "min_cluster_size": gb_mcs, "min_samples": gb_ms}
        return {"method": gb_method, "linkage": gb_linkage, "k": gb_k}

    # ── Methode waehlen ───────────────────────────────────────────────────────
    method_map = {
        "1": ("kmeans",       None),
        "2": ("hierarchical", "ward"),
        "3": ("hierarchical", "complete"),
        "4": ("hierarchical", "average"),
        "5": ("hierarchical", "single"),
        "6": ("hdbscan",      None),
    }
    while True:
        print("\n  Methode:")
        print("    [1] k-Means")
        print("    [2] Ward")
        print("    [3] Complete")
        print("    [4] Average")
        print("    [5] Single")
        print("    [6] HDBSCAN")
        m_choice = input("  Auswahl: ").strip()
        if m_choice in method_map:
            sel_method, sel_linkage = method_map[m_choice]
            break
        print("  Ungueltige Eingabe, bitte nochmal:")

    # ── HDBSCAN: beste Parameterkombination automatisch uebernehmen ───────────
    if sel_method == "hdbscan":
        hdb_rows = df_results[(df_results["method"] == "hdbscan") & (df_results["is_best"] == True)]
        if hdb_rows.empty:
            hdb_rows = df_results[df_results["method"] == "hdbscan"]
        if "rank_mean" in hdb_rows.columns and hdb_rows["rank_mean"].notna().any():
            best_hdb = hdb_rows.loc[hdb_rows["rank_mean"].idxmin()]
        else:
            best_hdb = hdb_rows.loc[hdb_rows["silhouette"].idxmax()]
        sel_mcs = int(best_hdb["min_cluster_size"])
        sel_ms  = None if pd.isna(best_hdb["min_samples"]) else int(best_hdb["min_samples"])
        sel_k   = int(best_hdb["n_clusters_found"])
        print(f"\n  Gewaehlt: hdbscan, min_cluster_size={sel_mcs}, min_samples={sel_ms}, k={sel_k} (automatisch)")
        return {"method": "hdbscan", "linkage": None, "k": sel_k,
                "min_cluster_size": sel_mcs, "min_samples": sel_ms}

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

    # ── Orientierungsplots vor der k-Auswahl anzeigen ─────────────────────────
    if df_features_z is not None and cfg is not None:
        import matplotlib.pyplot as plt
        from extension.viz_plots import elbow_plot as _elbow_plot

        # Elbow-Plot immer anzeigen (hilft bei k-Means und hierarchisch)
        _elbow_plot(df_results, df_features_z, cfg)
        _show_plot(cfg.OUTPUT_PLOTS_DIR / "elbow_plot.png")

        # Dendrogramm ohne Schnittlinie anzeigen (nur bei hierarchisch)
        if sel_method == "hierarchical":
            _show_dendrogram_preview(df_features_z, sel_linkage)

        print("\n  (Plots geschlossen - jetzt k waehlen)")

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
