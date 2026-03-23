# src/fatigue/io.py
#
# ZWECK: Alle Funktionen zum Einlesen und Erkunden von MAT-Dateien.
#
# ÄNDERUNGEN gegenüber alter Version:
#   - Alle Imports gesammelt AN DEN ANFANG (waren vorher doppelt und verstreut)
#   - StepFrequency aus extract_dual_axis_parameters entfernt (INT32_MAX = ungültig)
#   - Kommentare ergänzt damit der Code selbsterklärend ist

# ── Imports ──────────────────────────────────────────────────────────────────
# WARUM HIER: Alle Abhängigkeiten stehen am Anfang – das ist Python-Standard
# (PEP 8). Früher standen manche Imports mitten in der Datei, was verwirrend
# und fehleranfällig ist.
from pathlib import Path

import numpy as np
from scipy.io import loadmat
# ─────────────────────────────────────────────────────────────────────────────


def find_data_files(folder):
    """
    Findet alle unterstützten Messdateien in einem Ordner (rekursiv).

    Unterstützte Formate: .mat, .c3d, .csv, .txt

    Parameters
    ----------
    folder : str or Path

    Returns
    -------
    list of Path – sortiert
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {folder}")

    supported_extensions = [".mat", ".c3d", ".csv", ".txt"]

    files = [
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in supported_extensions
    ]

    return sorted(files)


def load_mat(path):
    """
    Lädt eine MATLAB .mat-Datei und gibt den Inhalt als dict zurück.

    WARUM squeeze_me=True: Entfernt überflüssige Dimensionen (z.B. Arrays
    der Form (1,1,N) werden zu (N,)).

    WARUM struct_as_record=False: MATLAB-Structs werden als Python-Objekte
    geladen, auf deren Felder man mit getattr() zugreifen kann.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"MAT-Datei nicht gefunden: {path}")

    mat = loadmat(path, squeeze_me=True, struct_as_record=False)

    # MATLAB-interne Metadaten entfernen – kein Messwert, nur Overhead
    mat.pop("__header__", None)
    mat.pop("__version__", None)
    mat.pop("__globals__", None)

    return mat


def get_parameters_r(mat):
    """
    Extrahiert PARAMETERS.R aus einem geladenen MAT-Dict.

    WARUM diese Funktion: Die Struktur PARAMETERS.R enthält alle
    berechneten Kennwerte (Winkel, Momente, Kontaktzeiten etc.).
    Der Zugriff ist je nach MATLAB-Version unterschiedlich (dict vs. Objekt),
    daher wird beides abgefangen.
    """
    if "PARAMETERS" not in mat:
        raise KeyError("Top-level Key 'PARAMETERS' nicht gefunden.")

    params = mat["PARAMETERS"]

    if isinstance(params, dict):
        if "R" not in params:
            raise KeyError("Key 'R' nicht gefunden in PARAMETERS.")
        return params["R"]

    if hasattr(params, "R"):
        return getattr(params, "R")

    raise TypeError(f"PARAMETERS hat unerwarteten Typ: {type(params)}")


def list_variable_names_from_r(r_obj):
    """
    Gibt alle Variablennamen in PARAMETERS.R zurück.
    """
    if isinstance(r_obj, dict):
        return sorted(r_obj.keys())

    if hasattr(r_obj, "__dict__"):
        return sorted(r_obj.__dict__.keys())

    if hasattr(r_obj, "dtype") and getattr(r_obj.dtype, "names", None):
        return sorted(list(r_obj.dtype.names))

    raise TypeError(f"R hat unerwarteten Typ: {type(r_obj)}")


def list_mat_variable_names(mat_path):
    """
    Lädt eine MAT-Datei und gibt Variablennamen aus PARAMETERS.R zurück.
    Nützlich zur schnellen Erkundung einer neuen Datei.
    """
    mat = load_mat(mat_path)
    r_obj = get_parameters_r(mat)
    return list_variable_names_from_r(r_obj)


def save_variable_names_to_txt(variable_names, output_path):
    """
    Speichert eine Liste von Variablennamen in eine Textdatei.
    """
    output_path = Path(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("Variablen in PARAMETERS.R:\n")
        f.write("---------------------------\n\n")
        for name in variable_names:
            f.write(name + "\n")


def export_mat_structure_to_txt(mat_path, output_txt):
    """
    Exportiert die Struktur einer MAT-Datei in eine Textdatei.
    Nützlich zur einmaligen Erkundung – nicht Teil des Analyse-Workflows.
    """
    mat = load_mat(mat_path)
    output_txt = Path(output_txt)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("MAT FILE STRUCTURE\n")
        f.write("===================\n\n")

        for key, obj in mat.items():
            f.write(f"TOP LEVEL: {key}\n")
            f.write("-----------------\n")
            f.write(f"Typ: {type(obj)}\n")

            if hasattr(obj, "shape"):
                f.write(f"Shape: {obj.shape}\n")

            if hasattr(obj, "__dict__"):
                f.write("Felder:\n")
                for subkey in obj.__dict__.keys():
                    subobj = getattr(obj, subkey)
                    f.write(f"  {subkey}\n")
                    f.write(f"    Typ: {type(subobj)}\n")
                    if hasattr(subobj, "shape"):
                        f.write(f"    Shape: {subobj.shape}\n")

            f.write("\n\n")


def export_top_level_fields_to_txt(mat_path, output_dir):
    """
    Exportiert für jede Top-Level-Variable die Feldnamen in separate TXT-Dateien.
    Beispiel: CONTACT_vars.txt, PARAMETERS_vars.txt
    Nützlich zur einmaligen Datenexploration.
    """
    mat_path = Path(mat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat = load_mat(mat_path)

    for top_key, obj in mat.items():
        fields = _get_field_names(obj)
        out_file = output_dir / f"{top_key}_vars.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"MAT-Datei: {mat_path.name}\n")
            f.write(f"TOP LEVEL: {top_key}\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Python-Typ: {type(obj)}\n")

            if hasattr(obj, "shape"):
                f.write(f"Shape: {obj.shape}\n")

            f.write("\nFELDER:\n")
            f.write("-" * 40 + "\n")

            if fields:
                for name in fields:
                    f.write(name + "\n")
            else:
                f.write("(keine Felder – möglicherweise ein Array/Skalar)\n")


def _get_field_names(obj) -> list:
    """
    Gibt Feldnamen eines MATLAB-ähnlichen Objekts zurück.

    WARUM PRIVAT (_): Diese Hilfsfunktion ist nur intern in io.py gebraucht
    und soll nicht von außen aufgerufen werden. Der Unterstrich signalisiert das.

    Unterstützt:
    - dict
    - scipy MATLAB-Structs (haben _fieldnames oder __dict__)
    - numpy structured arrays
    """
    if isinstance(obj, dict):
        return sorted([str(k) for k in obj.keys()])

    if hasattr(obj, "_fieldnames") and isinstance(getattr(obj, "_fieldnames"), list):
        return sorted([str(n) for n in obj._fieldnames if str(n) != "_fieldnames"])

    if hasattr(obj, "__dict__"):
        names = [k for k in obj.__dict__.keys() if not k.startswith("_")]
        return sorted([str(n) for n in names])

    if isinstance(obj, np.ndarray) and obj.dtype is not None and obj.dtype.names:
        return sorted([str(n) for n in obj.dtype.names])

    return []


def export_deep_variable_names(mat_path, output_dir):
    """
    Exportiert Variablennamen eine Ebene tiefer als export_top_level_fields_to_txt.
    Nützlich zur einmaligen Datenexploration.
    """
    mat = load_mat(mat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for top_key, obj in mat.items():
        if not hasattr(obj, "__dict__"):
            continue

        subkeys = list(obj.__dict__.keys())
        if not subkeys:
            continue

        sub_obj = getattr(obj, subkeys[0])

        if not hasattr(sub_obj, "__dict__"):
            continue

        var_names = sorted(sub_obj.__dict__.keys())
        out_file = output_dir / f"{top_key}_variables.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"{top_key} VARIABLEN\n")
            f.write("--------------------\n\n")
            f.write(f"Basierend auf: {subkeys[0]}\n\n")
            for v in var_names:
                f.write(v + "\n")


def export_real_variable_names(mat_path, output_dir):
    """
    Exportiert Variablennamen für jede Top-Level-Struktur.
    Nützlich zur einmaligen Datenexploration.
    """
    mat = load_mat(mat_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    for top_key, obj in mat.items():
        if not hasattr(obj, "__dict__"):
            continue

        level1_names = [n for n in obj.__dict__.keys() if n != "_fieldnames"]
        if not level1_names:
            continue

        level1_obj = getattr(obj, level1_names[0])

        if not hasattr(level1_obj, "__dict__"):
            continue

        var_names = sorted(
            [n for n in level1_obj.__dict__.keys() if n != "_fieldnames"]
        )

        out_file = output_dir / f"{top_key}_variables.txt"

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(f"TOP LEVEL: {top_key}\n")
            f.write(f"Basierend auf: {level1_names[0]}\n")
            f.write("\nVARIABLEN:\n\n")
            for v in var_names:
                f.write(v + "\n")


def extract_dual_axis_parameters(mat):
    """
    Extrahiert die für Dual-Axis benötigten Parameter aus CONTACT.

    WARUM NUR ContactTimes und FlightTimes:
    Das Feld StepFrequency in den MAT-Dateien enthält den Wert 2147483647
    (= INT32_MAX), was ein MATLAB-Platzhalter für "nicht berechnet" ist.
    Die Schrittfrequenz wird daher selbst aus ContactTimes und FlightTimes
    berechnet (siehe features.py: compute_step_frequency).

    WARUM trial_names[0]:
    Jede MAT-Datei enthält genau einen Trial (z.B. T_P61_km01_0).
    Falls zukünftig mehrere Trials pro Datei auftreten, muss diese
    Funktion erweitert werden – dann hier anpassen.

    Returns
    -------
    dict mit ContactTimes, FlightTimes (beides als numpy arrays)
    """
    contact = mat["CONTACT"]

    trial_names = [k for k in contact.__dict__.keys() if not k.startswith("_")]

    # Sicherheitscheck: mehr als ein Trial wäre unerwartet
    if len(trial_names) > 1:
        import warnings
        warnings.warn(
            f"Mehr als ein Trial gefunden: {trial_names}. "
            f"Es wird nur '{trial_names[0]}' verwendet. "
            f"Bitte Funktion extract_dual_axis_parameters prüfen.",
            UserWarning
        )

    trial = getattr(contact, trial_names[0])

    contact_times = np.asarray(getattr(trial, "ContactTimes"), dtype=float).flatten()
    flight_times = np.asarray(getattr(trial, "FlightTimes"), dtype=float).flatten()

    return dict(
        ContactTimes=contact_times,
        FlightTimes=flight_times,
    )


def parse_filename_info(mat_path):
    """
    Extrahiert Proband und Kilometer aus dem Dateinamen.

    Unterstützte Formate:
        P61_km01_0.mat  →  subject="P61", km=1.0
        P61_km01_5.mat  →  subject="P61", km=1.5
        P61_km10_0.mat  →  subject="P61", km=10.0

    WARUM diese Funktion: Der Dateiname ist die einzige Quelle für
    Proband-ID und km-Marke – sie sind nicht in der MAT-Datei selbst
    gespeichert.
    """
    name = Path(mat_path).stem
    parts = name.split("_")

    subject = parts[0]
    km_integer = parts[1].replace("km", "")
    km_decimal = parts[2]

    km = float(f"{int(km_integer)}.{km_decimal}")

    return dict(subject=subject, km=km)