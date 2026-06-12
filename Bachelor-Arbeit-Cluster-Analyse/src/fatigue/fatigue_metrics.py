# src/fatigue/fatigue_metrics.py
#
# ZWECK: Berechnung von Ermüdungsmetriken pro Proband aus dem Dual-Axis Dataset.
#
# KONZEPT:
#   Jeder Proband hat Messungen an den ganzzahligen km-Marken km 1 bis km 10.
#   Pro Proband werden zwei Kennwerte pro Variable (DF, SF_norm) berechnet:
#
#   Delta = Wert bei km 10 − Wert bei km 1
#   → Beschreibt die Gesamtveränderung von Start (km 1) bis Ende (km 10)
#   → Einfach interpretierbar, aber sensitiv gegenüber Ausreißern
#
#   Slope = Steigung der linearen Regression (km 1–10 → Variable)
#   → Beschreibt den Trend über alle 10 km
#   → Robuster als Delta, weil alle Messpunkte eingehen
#
#   WARUM BEIDE:
#   Delta und Slope messen leicht unterschiedliche Dinge.
#   Delta reagiert stark auf den letzten Messpunkt (z.B. Endspurt).
#   Slope ist stabiler aber kann durch Nicht-Linearität verzerrt sein.
#   Beide zusammen geben ein vollständigeres Bild der Ermüdung.
#   → Konsistent mit van Oeveren et al. (2021), die ebenfalls
#     Verlaufsparameter für die Klassifikation von Laufstilen nutzen.
#
# ÄNDERUNGEN gegenüber alter Version:
#   - Eingabe-Validierung ergänzt (mind. 2 km-Punkte nötig)
#   - Docstrings präzisiert (erwartete Spalten explizit genannt)
#   - Kommentare zu Delta vs. Slope ergänzt

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
# ─────────────────────────────────────────────────────────────────────────────


def compute_delta(series: pd.Series) -> float:
    """
    Berechnet die Differenz zwischen letztem und erstem Wert.

    Delta = letzter Wert − erster Wert

    INTERPRETATION:
    Positives Delta → Variable hat zugenommen (z.B. DF steigt → mehr Ermüdung)
    Negatives Delta → Variable hat abgenommen

    EINSCHRÄNKUNG:
    Delta nutzt nur zwei Punkte und ist sensitiv gegenüber Messausreißern
    am Start oder Ende. Für robustere Trendschätzung → compute_slope().

    Parameters
    ----------
    series : pd.Series – Zeitreihe einer Variable, sortiert nach km (km 1 bis km 10)

    Returns
    -------
    float – Wert bei km 10 minus Wert bei km 1
    """
    if len(series) < 2:
        raise ValueError(
            f"compute_delta benötigt mind. 2 Werte, erhalten: {len(series)}. "
            f"Proband hat möglicherweise zu wenige km-Messpunkte."
        )

    return float(series.iloc[-1] - series.iloc[0])


def compute_slope(x: np.ndarray, y: np.ndarray) -> float:
    """
    Berechnet die Steigung einer linearen Regression (x → y).

    Verwendet np.polyfit(x, y, 1) – kleinste Quadrate, Grad 1.

    INTERPRETATION:
    Slope_DF = +0.002 → DF steigt um 0.002 pro km
    Slope_SF = −0.01  → SF_norm sinkt um 0.01 pro km

    WARUM LINEARE REGRESSION:
    Lineare Regression nutzt alle Messpunkte und ist damit robuster
    als Delta. Voraussetzung: der Trend ist annähernd linear.
    Bei stark nicht-linearen Verläufen (z.B. starker Einbruch am Ende)
    kann die Steigung irreführend sein – dann Delta zusätzlich prüfen.

    Parameters
    ----------
    x : array-like – km-Werte (ganzzahlig: [1, 2, 3, ..., 10])
    y : array-like – Variablenwerte (DF oder SF_norm)

    Returns
    -------
    float – Steigung in Einheiten von y pro km
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) < 2:
        raise ValueError(
            f"compute_slope benötigt mind. 2 Punkte, erhalten: {len(x)}."
        )

    if len(x) != len(y):
        raise ValueError(
            f"x und y müssen gleich lang sein: len(x)={len(x)}, len(y)={len(y)}."
        )

    # np.polyfit gibt [slope, intercept] zurück – wir brauchen nur slope
    slope, _ = np.polyfit(x, y, 1)

    return float(slope)


def compute_subject_fatigue(df_subject: pd.DataFrame) -> dict:
    """
    Berechnet alle Ermüdungsmetriken für einen einzelnen Probanden.

    Eingabe-DataFrame muss folgende Spalten enthalten:
        km      – km-Marke (ganzzahlig: 1, 2, ..., 10)
        DF      – Duty Factor pro km-Marke
        SF_norm – Normierte Schrittfrequenz pro km-Marke

    WARUM SORTIERUNG:
    Die Daten werden nach km sortiert bevor Delta und Slope berechnet
    werden. Ohne Sortierung wären Delta und Slope potenziell falsch
    wenn die Dateien nicht in Reihenfolge eingelesen wurden.

    Parameters
    ----------
    df_subject : pd.DataFrame – Daten eines einzelnen Probanden (km 1–10)

    Returns
    -------
    dict mit:
        DF_start, DF_end      – DF bei km 1 / km 10
        Delta_DF              – Gesamtveränderung DF (km 10 − km 1)
        Slope_DF              – linearer Trend DF pro km (km 1–10)
        SF_start, SF_end      – SF_norm bei km 1 / km 10
        Delta_SF              – Gesamtveränderung SF_norm (km 10 − km 1)
        Slope_SF              – linearer Trend SF_norm pro km (km 1–10)
    """
    # Pflichtfelder prüfen
    required_cols = {"km", "DF", "SF_norm"}
    missing = required_cols - set(df_subject.columns)
    if missing:
        raise ValueError(
            f"df_subject fehlen Pflicht-Spalten: {missing}. "
            f"Vorhandene Spalten: {list(df_subject.columns)}"
        )

    # Nach km sortieren – wichtig für Delta (erster/letzter Wert)
    df_subject = df_subject.sort_values("km").reset_index(drop=True)

    if len(df_subject) < 2:
        raise ValueError(
            f"Proband hat nur {len(df_subject)} km-Messpunkt(e). "
            f"Mindestens 2 werden benötigt."
        )

    return dict(
        # Duty Factor
        DF_start  = float(df_subject["DF"].iloc[0]),
        DF_end    = float(df_subject["DF"].iloc[-1]),
        Delta_DF  = compute_delta(df_subject["DF"]),
        Slope_DF  = compute_slope(df_subject["km"], df_subject["DF"]),

        # Normierte Schrittfrequenz
        SF_start  = float(df_subject["SF_norm"].iloc[0]),
        SF_end    = float(df_subject["SF_norm"].iloc[-1]),
        Delta_SF  = compute_delta(df_subject["SF_norm"]),
        Slope_SF  = compute_slope(df_subject["km"], df_subject["SF_norm"]),
    )


def build_fatigue_feature_table(df_dual_axis: pd.DataFrame) -> pd.DataFrame:
    """
    Baut eine Feature-Tabelle mit einer Zeile pro Proband.

    Iteriert über alle Probanden im Dual-Axis Dataset und berechnet
    für jeden die Ermüdungsmetriken via compute_subject_fatigue().

    Eingabe-DataFrame muss folgende Spalten enthalten:
        Subject – Proband-ID (z.B. "P61")
        km      – km-Marke
        DF      – Duty Factor
        SF_norm – Normierte Schrittfrequenz

    Parameters
    ----------
    df_dual_axis : pd.DataFrame – langer Datensatz (eine Zeile pro Proband × km)

    Returns
    -------
    pd.DataFrame – breiter Datensatz (eine Zeile pro Proband)
        Spalten: Subject, DF_start, DF_end, Delta_DF, Slope_DF,
                 SF_start, SF_end, Delta_SF, Slope_SF
    """
    required_cols = {"Subject", "km", "DF", "SF_norm"}
    missing = required_cols - set(df_dual_axis.columns)
    if missing:
        raise ValueError(
            f"df_dual_axis fehlen Pflicht-Spalten: {missing}."
        )

    subjects = df_dual_axis["Subject"].unique()
    rows = []

    for s in subjects:
        df_s = df_dual_axis[df_dual_axis["Subject"] == s]
        metrics = compute_subject_fatigue(df_s)
        rows.append({"Subject": s, **metrics})

    return pd.DataFrame(rows)