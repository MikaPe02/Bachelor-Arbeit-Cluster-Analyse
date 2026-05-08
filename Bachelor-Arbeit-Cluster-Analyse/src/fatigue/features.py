# src/fatigue/features.py
#
# ZWECK: Berechnung der biomechanischen Kernparameter für das Dual-Axis Framework.
#
#   - Duty Factor (DF):       Anteil der Standphase an der Gesamtschrittzeit
#   - Schrittfrequenz (SF):   Schritte pro Sekunde
#   - SF_norm:                Normierte Schrittfrequenz nach van Oeveren et al. (2021)
#
# WISSENSCHAFTLICHE GRUNDLAGE:
#   van Oeveren et al. (2021): "The biomechanics of running and walking"
#   Normierung: SF_norm = SF * sqrt(l0 / g)
#   Beinlänge:  l0 = Trochanter-major-Abstand vom Boden
#   Schätzung:  l0 ≈ 0.53 × Körpergröße (De Leva, 1996)
#
# ÄNDERUNGEN gegenüber alter Version:
#   - Debug-print-Statements entfernt (gehören nicht in Produktionscode)
#   - Beinlängen-Platzhalter explizit mit TODO markiert
#   - estimate_leg_length() als wissenschaftlich begründeter Fallback eingebaut
#   - Kommentare ergänzt

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np

from fatigue.io import load_mat, extract_dual_axis_parameters
# ─────────────────────────────────────────────────────────────────────────────

# Erdbeschleunigung – Konstante, nie ändern
G = 9.81


# ── Beinlänge ────────────────────────────────────────────────────────────────

def estimate_leg_length(body_height_m: float) -> float:
    """
    Schätzt die funktionelle Beinlänge (Trochanter major → Boden).

    WARUM DIESE FUNKTION:
    Die Beinlänge ist für SF_norm zwingend nötig, aber nicht immer direkt
    gemessen. Falls nur die Körpergröße bekannt ist, liefert dieser Faktor
    eine wissenschaftlich anerkannte Näherung.

    Formel: l0 ≈ 0.53 × Körpergröße
    Quelle: De Leva (1996), Journal of Biomechanics, 29(9), 1223–1230.

    Parameters
    ----------
    body_height_m : float
        Körpergröße in Metern (z.B. 1.78)

    Returns
    -------
    float – geschätzte Beinlänge in Metern

    Beispiel
    --------
    >>> estimate_leg_length(1.78)
    0.9434
    """
    # Faktor 0.53 nach De Leva (1996) – Verhältnis Beinlänge zu Körpergröße
    LEG_LENGTH_FACTOR = 0.53
    return float(body_height_m * LEG_LENGTH_FACTOR)


# ── Kernberechnungen ─────────────────────────────────────────────────────────

def compute_duty_factor(contact_times, flight_times):
    """
    Berechnet den mittleren Duty Factor über alle Schritte.

    Definition: DF = Standzeit / (Standzeit + Flugzeit)

    WARUM MITTELWERT:
    Jede MAT-Datei enthält ~88 Schritte (ca. 500m bei km-Marken).
    Wir mitteln über alle Schritte um einen repräsentativen Wert
    pro km-Abschnitt zu erhalten – konsistent mit van Oeveren et al.

    WARUM min(len(ct), len(ft)):
    ContactTimes und FlightTimes können sich um 1 Element unterscheiden
    (letzter Schritt ohne abschließende Flugzeit). Wir schneiden auf
    gleiche Länge um paarweise zu rechnen.

    Parameters
    ----------
    contact_times : array-like – Standzeiten in Sekunden
    flight_times  : array-like – Flugzeiten in Sekunden

    Returns
    -------
    float – mittlerer Duty Factor (dimensionslos, typisch 0.55–0.70)
    """
    ct = np.asarray(contact_times, dtype=float).flatten()
    ft = np.asarray(flight_times, dtype=float).flatten()

    # Auf gleiche Länge kürzen
    n = min(len(ct), len(ft))
    ct = ct[:n]
    ft = ft[:n]

    # ÄNDERUNG: print-Statements entfernt – diese waren Debug-Ausgaben
    # und haben bei jedem Funktionsaufruf die Konsole geflutet.
    # Für Debugging: temporär wieder einkommentieren.
    # print("ContactTimes:", len(ct))
    # print("FlightTimes:", len(ft))
    # print("Used strides:", n)

    df = ct / (ct + ft)

    # Ungültige Werte (Division durch 0, NaN, Inf) entfernen
    df = df[np.isfinite(df)]

    if df.size == 0:
        raise ValueError(
            "Duty Factor Array ist leer nach Bereinigung. "
            "Bitte ContactTimes und FlightTimes prüfen."
        )

    return float(np.mean(df))


def compute_step_frequency(contact_times, flight_times):
    """
    Berechnet die mittlere Schrittfrequenz aus Kontakt- und Flugzeiten.

    Definition: SF = 1 / (Kontaktzeit + Flugzeit)

    WARUM NICHT DEN VORBERECHNETEN WERT AUS DER MAT-DATEI:
    Das Feld StepFrequency in den MAT-Dateien enthält den Wert 2147483647
    (INT32_MAX) – ein MATLAB-Platzhalter für "nicht berechnet".
    Daher wird SF hier selbst berechnet. Validierung hat ergeben:
    SF ≈ 2.999 Hz, was für Laufen bei ~10 km/h plausibel ist.

    Parameters
    ----------
    contact_times : array-like – Standzeiten in Sekunden
    flight_times  : array-like – Flugzeiten in Sekunden

    Returns
    -------
    float – mittlere Schrittfrequenz in Hz (Schritte/Sekunde)
    """
    ct = np.asarray(contact_times, dtype=float).flatten()
    ft = np.asarray(flight_times, dtype=float).flatten()

    n = min(len(ct), len(ft))
    ct = ct[:n]
    ft = ft[:n]

    stride_time = ct + ft

    # Schutz vor Division durch 0
    sf = 1.0 / stride_time[stride_time > 0]

    if sf.size == 0:
        raise ValueError("Keine gültigen Schrittzeiten gefunden.")

    return float(np.mean(sf))


def compute_sf_norm(contact_times, flight_times, leg_length_m):
    """
    Berechnet die normierte Schrittfrequenz nach van Oeveren et al. (2021).

    Formel: SF_norm = SF × sqrt(l0 / g)

    WARUM NORMIEREN:
    Die Normierung macht SF unabhängig von der Beinlänge und erlaubt
    den Vergleich zwischen Probanden unterschiedlicher Körpergröße.
    SF_norm ist dimensionslos und typischerweise im Bereich 0.55–0.95.

    Parameters
    ----------
    contact_times : array-like
    flight_times  : array-like
    leg_length_m  : float – Beinlänge in Metern (Trochanter major → Boden)
                    Schätzung: estimate_leg_length(Körpergröße_in_m)

    Returns
    -------
    float – normierte Schrittfrequenz (dimensionslos)
    """
    if leg_length_m <= 0:
        raise ValueError(
            f"Beinlänge muss positiv sein, erhalten: {leg_length_m}. "
            f"Platzhalter? → estimate_leg_length() verwenden."
        )

    SF = compute_step_frequency(contact_times, flight_times)
    SF_norm = SF * np.sqrt(leg_length_m / G)

    return float(SF_norm)


def dual_axis_from_mat(mat_path, leg_length_m):
    """
    Berechnet DF und SF_norm direkt aus einer MAT-Datei.

    Dies ist die Hauptfunktion für den Dual-Axis Workflow.
    Sie kombiniert load_mat → extract_dual_axis_parameters →
    compute_duty_factor → compute_sf_norm.

    Parameters
    ----------
    mat_path     : str or Path – Pfad zur MAT-Datei
    leg_length_m : float – Beinlänge in Metern
                   # TODO: Platzhalter 1.0 ersetzen sobald Daten vorliegen
                   Fallback: estimate_leg_length(Körpergröße_in_m)

    Returns
    -------
    dict mit:
        DF      – Duty Factor (dimensionslos)
        SF_norm – Normierte Schrittfrequenz (dimensionslos)
    """
    mat = load_mat(mat_path)
    p = extract_dual_axis_parameters(mat)

    DF = compute_duty_factor(p["ContactTimes"], p["FlightTimes"])
    SF_norm = compute_sf_norm(p["ContactTimes"], p["FlightTimes"], leg_length_m)

    return dict(DF=DF, SF_norm=SF_norm)