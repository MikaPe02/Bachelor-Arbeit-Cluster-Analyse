# src/fatigue/preprocessing.py
#
# ZWECK: Vorverarbeitungsschritte fuer die Fatigue-Analyse-Pipeline.
#
# FUNKTIONEN:
#   z_transform(df_features) -> pd.DataFrame
#       Z-Transformation aller numerischen Feature-Spalten.
#
#   sanity_check(df) -> None
#       Prueft ob DF und SF_norm sich zwischen km-Marken unterscheiden.

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler


def z_transform(df_features: pd.DataFrame) -> pd.DataFrame:
    """Z-Transformation aller numerischen Feature-Spalten.

    Parameters
    ----------
    df_features : pd.DataFrame
        Feature-Tabelle mit Spalte 'Subject' + numerischen Feature-Spalten.

    Returns
    -------
    pd.DataFrame – z-transformierte Tabelle, gleiche Struktur wie Input.
    """
    print("\n=== Schritt 3: Z-Transformation ===")

    feature_cols = [c for c in df_features.columns if c != "Subject"]

    scaler = StandardScaler()
    X_z    = scaler.fit_transform(df_features[feature_cols])

    df_z = pd.DataFrame(X_z, columns=feature_cols)
    df_z.insert(0, "Subject", df_features["Subject"].values)

    means = df_features[feature_cols].mean()
    stds  = df_features[feature_cols].std()
    print("  Mittelwerte (vor Z-Trafo):")
    for col in feature_cols:
        print(f"    {col:15s}: mean={means[col]:+.4f}, std={stds[col]:.4f}")

    return df_z


def sanity_check(df: pd.DataFrame) -> None:
    """Prueft ob DF und SF_norm sich zwischen km-Marken unterscheiden.

    Berechnet pro Proband den Wertebereich (max - min) von DF und SF_norm.
    Probanden mit Range = 0 veraendern sich nie → verdaechtig, evtl. Datenfehler.

    Parameters
    ----------
    df : pd.DataFrame
        Langer Datensatz mit Spalten 'Subject', 'DF', 'SF_norm', 'km'.
    """
    print("\n=== Schritt 7: Sanity-Check ===")

    for var in ("DF", "SF_norm"):
        ranges = df.groupby("Subject")[var].agg(lambda s: s.max() - s.min())
        flat   = ranges[ranges == 0.0]

        print(f"\n  {var}:")
        print(f"    Median Range: {ranges.median():.6f}")
        print(f"    Min Range:    {ranges.min():.6f}  (Proband: {ranges.idxmin()})")
        print(f"    Max Range:    {ranges.max():.6f}  (Proband: {ranges.idxmax()})")

        if len(flat) > 0:
            print(f"    WARNUNG: {len(flat)} Proband(en) mit {var}-Range = 0: {list(flat.index)}")
        else:
            print(f"    OK: Alle Probanden zeigen Variation in {var}")
