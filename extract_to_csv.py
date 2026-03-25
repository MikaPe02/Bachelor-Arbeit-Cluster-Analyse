# extract_to_csv.py
#
# ZWECK: Extrahiert DF und SF_norm aus allen MAT-Dateien und speichert
#        das Ergebnis als zentrale CSV-Datei.
#
# WANN AUSFÜHREN:
#   - Einmalig zu Beginn des Projekts
#   - Erneut wenn neue Probanden-Daten hinzukommen
#
# INPUT:  MAT-Dateien in DATA_RAW_FOLDER (lokal, nicht in Git)
# OUTPUT: data/processed/dual_axis_dataset.csv
#
# BEINLÄNGE:
#   Aktuell Platzhalter (1.0 m) für alle Probanden.
#   Sobald echte Werte vorliegen: in data/subjects.csv eintragen.
#   Das Skript liest subjects.csv automatisch und verwendet echte Werte
#   wenn vorhanden, sonst Schätzung via De Leva (1996): l0 = 0.53 × Größe.
#   Falls beides fehlt: Platzhalter 1.0 m mit Warnung.

# ── Imports ──────────────────────────────────────────────────────────────────
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

import pandas as pd

# Projektpfad setzen
project_root = Path(__file__).resolve().parent
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.io import find_data_files, load_mat, extract_dual_axis_parameters, parse_filename_info
from fatigue.features import compute_duty_factor, compute_sf_norm, estimate_leg_length
# ─────────────────────────────────────────────────────────────────────────────


# ── Konfiguration ─────────────────────────────────────────────────────────────
# Output
OUTPUT_FILE = project_root / "data" / "processed" / "dual_axis_dataset.csv"

# Probanden-Metadaten (Beinlängen)
SUBJECTS_FILE = project_root / "data" / "subjects.csv"

# Fallback falls keine Beinlänge bekannt
LEG_LENGTH_FALLBACK = 1.0  # TODO: ersetzen sobald Daten vorliegen
# ─────────────────────────────────────────────────────────────────────────────


def load_subject_metadata(subjects_file: Path) -> dict:
    """
    Lädt Probanden-Metadaten aus subjects.csv.

    Priorität der Beinlänge:
    1. leg_length_m direkt gemessen
    2. body_height_m vorhanden -> schaetzen via De Leva (1996)
    3. Beides fehlt -> Fallback-Platzhalter mit Warnung
    """
    if not subjects_file.exists():
        print(f"WARNUNG: subjects.csv nicht gefunden: {subjects_file}")
        print("  Platzhalter-Beinlaenge wird fuer alle Probanden verwendet.")
        return {}

    # Leere Felder explizit als NaN einlesen
    df = pd.read_csv(subjects_file, na_values=["", " ", "NA", "nan"])

    leg_lengths = {}

    for _, row in df.iterrows():
        subject = str(row["Subject"]).strip()

        # Leere oder ungueltige Subject-IDs ueberspringen
        if not subject or subject == "nan":
            continue

        # Hilfsfunktion: prueft ob Wert eine nutzbare Zahl ist
        def is_valid(val):
            try:
                return pd.notna(val) and float(val) > 0
            except (TypeError, ValueError):
                return False

        # Prioritaet 1: direkt gemessene Beinlaenge
        if is_valid(row.get("leg_length_m")):
            leg_lengths[subject] = float(row["leg_length_m"])

        # Prioritaet 2: Schaetzung aus Koerpergroesse
        elif is_valid(row.get("body_height_m")):
            estimated = estimate_leg_length(float(row["body_height_m"]))
            leg_lengths[subject] = estimated
            print(f"  {subject}: Beinlaenge geschaetzt "
                  f"({row['body_height_m']} m x 0.53 = {estimated:.3f} m)")

        # Prioritaet 3: Platzhalter
        else:
            print(f"  WARNUNG {subject}: Keine Beinlaenge bekannt "
                  f"-> Platzhalter {LEG_LENGTH_FALLBACK} m")
            leg_lengths[subject] = LEG_LENGTH_FALLBACK

    return leg_lengths


def extract_all(data_folder: str, leg_lengths: dict) -> pd.DataFrame:
    """
    Liest alle MAT-Dateien und berechnet DF + SF_norm pro Datei.

    Parameters
    ----------
    data_folder : str – Pfad zum Ordner mit MAT-Dateien
    leg_lengths : dict – {subject_id: leg_length_m}

    Returns
    -------
    pd.DataFrame mit Spalten: Subject, km, DF, SF_norm, leg_length_m
    """
    files = find_data_files(data_folder)
    mat_files = [f for f in files if f.suffix.lower() == ".mat"]

    if not mat_files:
        raise SystemExit(f"Keine MAT-Dateien gefunden in: {data_folder}")

    print(f"\nGefundene MAT-Dateien: {len(mat_files)}")
    print("-" * 50)

    rows = []
    errors = []

    for mat_path in mat_files:
        try:
            # Proband + km aus Dateiname
            info = parse_filename_info(mat_path)
            subject = info["subject"]
            km = info["km"]

            # Beinlänge bestimmen
            leg_length = leg_lengths.get(subject, LEG_LENGTH_FALLBACK)
            if subject not in leg_lengths:
                print(f"  WARNUNG {mat_path.name}: Proband '{subject}' nicht in subjects.csv -> Platzhalter")

            # MAT laden und Parameter extrahieren
            mat = load_mat(mat_path)
            p = extract_dual_axis_parameters(mat)

            # DF und SF_norm berechnen
            df_val = compute_duty_factor(p["ContactTimes"], p["FlightTimes"])
            sf_norm = compute_sf_norm(p["ContactTimes"], p["FlightTimes"], leg_length)

            rows.append(dict(
                Subject      = subject,
                km           = km,
                DF           = round(df_val, 6),
                SF_norm      = round(sf_norm, 6),
                leg_length_m = leg_length,
            ))

            print(f"  OK {mat_path.name:<30} DF={df_val:.4f}  SF_norm={sf_norm:.4f}")

        except Exception as e:
            print(f"  FEHLER {mat_path.name}: {e}")
            errors.append((mat_path.name, str(e)))

    print("-" * 50)
    print(f"Erfolgreich: {len(rows)} | Fehler: {len(errors)}")

    if errors:
        print("\nFehlerhafte Dateien:")
        for name, err in errors:
            print(f"  {name}: {err}")

    return pd.DataFrame(rows)


# == Hauptprogramm ===========================================================
if __name__ == "__main__":

    print("=" * 50)
    print("EXTRAKTION: MAT nach CSV")
    print("=" * 50)

    # Schritt 1: Beinlaengen laden
    print("\nLade Probanden-Metadaten...")
    leg_lengths = load_subject_metadata(SUBJECTS_FILE)
    print(f"Bekannte Probanden: {len(leg_lengths)}")

    # Schritt 2: Ordner mit MAT-Dateien auswaehlen
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    data_raw_folder = filedialog.askdirectory(
        title="Ordner mit MAT-Dateien auswaehlen",
        parent=root,
    )
    root.destroy()
    if not data_raw_folder:
        raise SystemExit("Kein Ordner ausgewaehlt - Programm wird beendet.")
    print(f"\nAusgewaehlter Ordner: {data_raw_folder}")

    # Schritt 3: Alle MAT-Dateien extrahieren
    df = extract_all(data_raw_folder, leg_lengths)

    if df.empty:
        raise SystemExit("Keine Daten extrahiert - Programm wird beendet.")

    # Schritt 4: Sortieren nach Proband und km
    df = df.sort_values(["Subject", "km"]).reset_index(drop=True)

    # Schritt 5: Speichern
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\nGespeichert: {OUTPUT_FILE}")
    print(f"  {len(df)} Zeilen | {df['Subject'].nunique()} Proband(en)")
    print("\nVorschau:")
    print(df.head(10).to_string(index=False))