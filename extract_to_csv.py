# extract_to_csv.py
#
# ZWECK: Extrahiert DF und SF_norm aus allen MAT-Dateien und speichert
#        das Ergebnis als zentrale CSV-Datei.
#        Enthält außerdem extract_subjects_from_xlsx() zum Befüllen von
#        data/subjects.csv aus der Messprotokoll-XLSX.
#
# WANN AUSFÜHREN:
#   - Einmalig zu Beginn des Projekts
#   - Erneut wenn neue Probanden-Daten hinzukommen
#
# INPUT:  MAT-Dateien in DATA_RAW_FOLDER (lokal, nicht in Git)
# OUTPUT: data/processed/dual_axis_dataset.csv
#         data/subjects.csv  (via extract_subjects_from_xlsx)
#
# BEINLÄNGE:
#   Werte kommen aus data/subjects.csv (leg_length_m = Mittelwert li+re).
#   Fallback: Schätzung via De Leva (1996): l0 = 0.53 × Größe.
#   Falls beides fehlt: Platzhalter 1.0 m mit Warnung.

# ── Imports ──────────────────────────────────────────────────────────────────
import re
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

# Probanden-Metadaten
SUBJECTS_FILE   = project_root / "data" / "subjects.csv"
XLSX_PROTOCOLS  = project_root / "data" / "Raw_Data" / "Uebersicht_Messprotokolle_2026_04_15.xlsx"

# Fallback falls keine Beinlänge bekannt
LEG_LENGTH_FALLBACK = 1.0

# Suffixe die eine Zusatzmessung kennzeichnen (werden iterativ entfernt)
_SUBJECT_SUFFIX_RE = re.compile(
    r"(?i)(Estrogen|Menstruation|ohneFatigue|_?bis_?\d+km\d*|schnelleGeschw)\s*$"
)

def _clean_subject_id(raw: str) -> str:
    s = str(raw).strip()
    while True:
        cleaned = _SUBJECT_SUFFIX_RE.sub("", s).strip()
        if cleaned == s:
            break
        s = cleaned
    return s

# Borg-Spalten - bei NaN in einer dieser Spalten gilt die Zeile als unvollständig
_BORG_COLS = ["Borg_pre", "Borg_2km", "Borg_4km", "Borg_6km", "Borg_8km", "Borg_post"]
# ─────────────────────────────────────────────────────────────────────────────


def extract_subjects_from_xlsx(
    xlsx_path: Path = XLSX_PROTOCOLS,
    output_path: Path = SUBJECTS_FILE,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Liest Probanden-Metadaten aus der Messprotokoll-XLSX und schreibt
    sie in data/subjects.csv.

    Logik:
    - Subject-ID: Suffixe (Estrogen, Menstruation, ohneFatigue, bis5km …)
      werden entfernt -> nur 'P01'-Stil
    - Körpergröße: Werte > 3.0 sind in cm eingetragen -> durch 100 teilen
    - Beinlänge li/re: immer in cm -> durch 100 teilen
    - leg_length_m: Mittelwert aus li und re
    - Duplikate (gleiche bereinigte Subject-ID):
        1. Zeile mit NaN in einem Borg-Wert wird verworfen
        2. Bei Gewicht-Unterschied: kleinsten Wert nehmen
        3. Alle anderen relevanten Felder müssen identisch sein -
           sonst Warnung und erste Zeile behalten

    Returns
    -------
    df : pd.DataFrame mit Spalten:
        Subject, body_height_m, dominant_leg,
        leg_length_li_m, leg_length_re_m, leg_length_m,
        weight_kg, speed_ms
    selected_raw_names : dict {bereinigteID -> originalerProbandName}
        z.B. {"P76": "P76Menstruation"} — zeigt welcher Ordner/Lauf gewählt wurde
    """
    print(f"\nLese XLSX: {xlsx_path.name}")
    raw = pd.read_excel(xlsx_path)

    # --- Relevante Spalten auswählen -----------------------------------------
    keep = [
        "Proband", "Geschlecht", "Groesse", "DominantesBein",
        "Beinlaenge_li", "Beinlaenge_re",
        "Gewicht_preRun", "Geschwindigkeit",
    ] + _BORG_COLS
    df = raw[keep].copy()

    # --- Subject-ID bereinigen ------------------------------------------------
    df["Subject"] = df["Proband"].apply(_clean_subject_id)

    # --- Einheiten korrigieren ------------------------------------------------
    # Körpergröße: Werte > 3.0 sind in cm
    df["body_height_m"] = df["Groesse"].apply(
        lambda v: v / 100.0 if pd.notna(v) and v > 3.0 else v
    )
    # Beinlänge: immer cm -> m
    df["leg_length_li_m"] = df["Beinlaenge_li"].apply(
        lambda v: round(v / 100.0, 4) if pd.notna(v) else v
    )
    df["leg_length_re_m"] = df["Beinlaenge_re"].apply(
        lambda v: round(v / 100.0, 4) if pd.notna(v) else v
    )
    df["leg_length_m"] = df[["leg_length_li_m", "leg_length_re_m"]].mean(axis=1).round(4)

    df.rename(columns={
        "Geschlecht":     "sex",
        "DominantesBein": "dominant_leg",
        "Gewicht_preRun": "weight_kg",
        "Geschwindigkeit": "speed_ms",
    }, inplace=True)

    # --- Duplikate auflösen ---------------------------------------------------
    result_rows: list[dict] = []
    selected_raw_names: dict[str, str] = {}  # {bereinigteID -> gewählter Proband-Name}
    compare_cols = ["body_height_m", "dominant_leg",
                    "leg_length_li_m", "leg_length_re_m", "speed_ms"]

    for subject, group in df.groupby("Subject", sort=False):
        if len(group) == 1:
            chosen = group.iloc[0]
            result_rows.append(_row_to_dict(chosen))
            selected_raw_names[subject] = str(chosen["Proband"]).strip()
            continue

        # Zeilen mit unvollständigen Borg-Werten verwerfen
        complete = group[group[_BORG_COLS].notna().all(axis=1)]
        if len(complete) == 0:
            # alle unvollständig -> erste behalten, Warnung
            print(f"  WARNUNG {subject}: Alle Einträge haben fehlende Borg-Werte "
                  f"- erster Eintrag ({group.iloc[0]['Proband']}) wird behalten.")
            complete = group.iloc[[0]]
        elif len(complete) < len(group):
            dropped = group[~group.index.isin(complete.index)]["Proband"].tolist()
            print(f"  {subject}: Einträge mit fehlenden Borg-Werten entfernt: {dropped}")

        group = complete

        if len(group) == 1:
            chosen = group.iloc[0]
            result_rows.append(_row_to_dict(chosen))
            selected_raw_names[subject] = str(chosen["Proband"]).strip()
            continue

        # Prüfe ob nicht-Gewicht-Felder identisch sind
        ref = group.iloc[0]
        for col in compare_cols:
            vals = group[col].dropna().unique()
            if len(vals) > 1:
                print(f"  WARNUNG {subject}: Spalte '{col}' unterscheidet sich "
                      f"zwischen Einträgen {group['Proband'].tolist()} -> "
                      f"Werte: {vals.tolist()} - erster Eintrag wird behalten.")

        # Kleinstes Gewicht nehmen
        min_weight = group["weight_kg"].min()
        row = _row_to_dict(ref)
        row["weight_kg"] = min_weight
        result_rows.append(row)
        selected_raw_names[subject] = str(ref["Proband"]).strip()

    # --- Ausgabe-DataFrame zusammenbauen ------------------------------------
    out_cols = [
        "Subject", "sex", "body_height_m", "dominant_leg",
        "leg_length_li_m", "leg_length_re_m", "leg_length_m",
        "weight_kg", "speed_ms",
    ]
    out = pd.DataFrame(result_rows)[out_cols].sort_values("Subject").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"  -> {len(out)} Probanden gespeichert in {output_path}")

    # Übersicht welcher Lauf pro Proband gewählt wurde
    print("\n  Gewählter Lauf pro Proband:")
    for clean_id, raw_name in sorted(selected_raw_names.items()):
        if clean_id != raw_name:
            print(f"    {clean_id} <- {raw_name}")

    return out, selected_raw_names


def _row_to_dict(row: pd.Series) -> dict:
    return {
        "Subject":         row["Subject"],
        "sex":             row["sex"],
        "body_height_m":   row["body_height_m"],
        "dominant_leg":    row["dominant_leg"],
        "leg_length_li_m": row["leg_length_li_m"],
        "leg_length_re_m": row["leg_length_re_m"],
        "leg_length_m":    row["leg_length_m"],
        "weight_kg":       row["weight_kg"],
        "speed_ms":        row["speed_ms"],
    }


def load_subject_metadata(subjects_file: Path) -> tuple[dict, dict]:
    """
    Lädt Probanden-Metadaten aus subjects.csv.

    Priorität der Beinlänge:
    1. leg_length_m direkt gemessen
    2. body_height_m vorhanden -> schaetzen via De Leva (1996)
    3. Beides fehlt -> Fallback-Platzhalter mit Warnung

    Returns
    -------
    tuple: (leg_lengths, speeds)
        leg_lengths : dict {subject_id: leg_length_m}
        speeds      : dict {subject_id: speed_ms} - nur wenn Wert vorhanden
    """
    if not subjects_file.exists():
        print(f"WARNUNG: subjects.csv nicht gefunden: {subjects_file}")
        print("  Platzhalter-Beinlaenge wird fuer alle Probanden verwendet.")
        return {}, {}

    # Leere Felder explizit als NaN einlesen
    df = pd.read_csv(subjects_file, na_values=["", " ", "NA", "nan"])

    leg_lengths = {}
    speeds = {}

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

        # Speed (optional) - nur wenn Spalte vorhanden und Wert gültig
        if is_valid(row.get("speed_ms")):
            speeds[subject] = float(row["speed_ms"])

    return leg_lengths, speeds


def extract_all(
    data_folder: str | Path,
    leg_lengths: dict,
    speeds: dict,
    selected_raw_names: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Liest MAT-Dateien und berechnet DF + SF_norm pro Datei.

    Wenn selected_raw_names übergeben wird, werden pro Proband nur die
    MAT-Dateien aus dem passenden Unterordner gelesen (z.B. 'P76Menstruation').
    Ohne selected_raw_names werden alle MAT-Dateien im data_folder gelesen.

    Parameters
    ----------
    data_folder        : Pfad zum Ordner mit Probanden-Unterordnern (oder MAT-Dateien)
    leg_lengths        : {subject_id: leg_length_m}
    speeds             : {subject_id: speed_ms}
    selected_raw_names : {bereinigte_id -> ordnername} z.B. {"P76": "P76Menstruation"}

    Returns
    -------
    pd.DataFrame mit Spalten: Subject, km, DF, SF_norm, leg_length_m, speed_ms
    """
    data_folder = Path(data_folder)

    # MAT-Dateien sammeln: entweder gezielt aus Unterordnern oder alle rekursiv
    if selected_raw_names:
        mat_files = []
        for clean_id, raw_name in selected_raw_names.items():
            subject_dir = data_folder / raw_name
            if subject_dir.is_dir():
                found = list(subject_dir.rglob("*.mat"))
                mat_files.extend(found)
            else:
                # Kein eigener Unterordner -> direkt im data_folder suchen
                found = list(data_folder.glob(f"{raw_name}_km*.mat"))
                mat_files.extend(found)
                if not found:
                    print(f"  WARNUNG: Kein Ordner und keine Dateien für '{raw_name}' gefunden.")
    else:
        files = find_data_files(data_folder)
        mat_files = [f for f in files if f.suffix.lower() == ".mat"]

    if not mat_files:
        raise SystemExit(f"Keine MAT-Dateien gefunden in: {data_folder}")

    print(f"\nGefundene MAT-Dateien: {len(mat_files)}")
    print("-" * 50)

    # Umgekehrtes Mapping: ordnername -> bereinigte_id (für Umbenennung)
    raw_to_clean: dict[str, str] = {}
    if selected_raw_names:
        raw_to_clean = {v: k for k, v in selected_raw_names.items()}

    rows = []
    errors = []

    for mat_path in sorted(mat_files):
        try:
            info = parse_filename_info(mat_path)
            raw_subject = info["subject"]   # z.B. "P76Menstruation" (aus Dateiname)
            km = info["km"]

            # Bereinigten Subject-Namen bestimmen
            # Zuerst über Ordner-Mapping, sonst direkt bereinigen
            parent_name = mat_path.parent.name
            if parent_name in raw_to_clean:
                subject = raw_to_clean[parent_name]
            else:
                subject = _clean_subject_id(raw_subject)

            # Beinlänge bestimmen
            leg_length = leg_lengths.get(subject, LEG_LENGTH_FALLBACK)
            if subject not in leg_lengths:
                print(f"  WARNUNG {mat_path.name}: '{subject}' nicht in subjects.csv -> Platzhalter")

            # MAT laden und Parameter extrahieren
            mat = load_mat(mat_path)
            p = extract_dual_axis_parameters(mat)

            df_val = compute_duty_factor(p["ContactTimes"], p["FlightTimes"])
            sf_norm = compute_sf_norm(p["ContactTimes"], p["FlightTimes"], leg_length)

            rows.append(dict(
                Subject = subject,
                km      = km,
                DF      = round(df_val, 6),
                SF_norm = round(sf_norm, 6),
            ))

            print(f"  OK {mat_path.name:<35} Subject={subject}  DF={df_val:.4f}  SF_norm={sf_norm:.4f}")

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

    # Schritt 0: subjects.csv aus XLSX aktualisieren
    # Gibt auch selected_raw_names zurück: {bereinigte_id -> gewählter Ordnername}
    selected_raw_names: dict[str, str] = {}
    if XLSX_PROTOCOLS.exists():
        print("\nSchritt 0: Probanden-Metadaten aus XLSX extrahieren...")
        _, selected_raw_names = extract_subjects_from_xlsx(XLSX_PROTOCOLS, SUBJECTS_FILE)
    else:
        print(f"\nSchritt 0: XLSX nicht gefunden ({XLSX_PROTOCOLS.name}) - subjects.csv wird nicht aktualisiert.")

    # Schritt 1: Beinlaengen und Speed laden
    print("\nLade Probanden-Metadaten...")
    leg_lengths, speeds = load_subject_metadata(SUBJECTS_FILE)
    print(f"Bekannte Probanden: {len(leg_lengths)}")
    print(f"Probanden mit Speed: {len(speeds)}")

    # Schritt 2: Daten-Ordner bestimmen
    import config as _cfg
    if _cfg.DATA_RAW_FOLDER and Path(_cfg.DATA_RAW_FOLDER).is_dir():
        data_raw_folder = Path(_cfg.DATA_RAW_FOLDER)
        print(f"\nDaten-Ordner (aus config.py): {data_raw_folder}")
    else:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        chosen = filedialog.askdirectory(
            title="Ordner mit MAT-Dateien auswaehlen",
            parent=root,
        )
        root.destroy()
        if not chosen:
            raise SystemExit("Kein Ordner ausgewaehlt - Programm wird beendet.")
        data_raw_folder = Path(chosen)
        print(f"\nAusgewaehlter Ordner: {data_raw_folder}")

    # Schritt 3: MAT-Dateien extrahieren (nur gewählte Läufe pro Proband)
    df = extract_all(data_raw_folder, leg_lengths, speeds, selected_raw_names or None)

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