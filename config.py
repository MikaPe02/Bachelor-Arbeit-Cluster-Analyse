# config.py
#
# ZWECK: Zentrale Konfiguration fuer die gesamte Analyse.
#        Alle Einstellungen die sich zwischen Projekten oder
#        Rechnern unterscheiden koennen stehen hier.
#
# ANPASSEN:
#   - DATA_RAW_FOLDER: Pfad zu den MAT-Dateien auf deinem Rechner
#   - Clustering-Parameter nach Bedarf
#   - Alles andere bleibt unveraendert
#
# BEINLAENGE:
#   Wird automatisch aus data/subjects.csv gelesen.
#   Prioritaet: leg_length_m (direkt) > body_height_m (Schaetzung) > Fallback
#   Quelle Schaetzung: De Leva (1996), Faktor 0.53 x Koerpergroesse

from pathlib import Path

# ── Projektpfade ─────────────────────────────────────────────────────────────

# Wurzelverzeichnis des Projekts (dort wo config.py liegt)
PROJECT_ROOT = Path(__file__).resolve().parent

# Pfad zu den MAT-Rohdaten – wird per Ordnerauswahl in extract_to_csv.py gesetzt
DATA_RAW_FOLDER = None

# Verarbeitete Daten (werden von extract_to_csv.py erzeugt)
DATA_PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
DUAL_AXIS_CSV       = DATA_PROCESSED_DIR / "dual_axis_dataset.csv"

# Probanden-Metadaten (Beinlaenge / Koerpergroesse)
# Spalten: Subject, leg_length_m, body_height_m, notes
SUBJECTS_CSV = PROJECT_ROOT / "data" / "subjects.csv"

# Ausgabe-Ordner
OUTPUT_DIR       = PROJECT_ROOT / "Outputs"
OUTPUT_PLOTS_DIR = OUTPUT_DIR / "Plots"
OUTPUT_DATA_DIR  = OUTPUT_DIR / "Data"


# ── Beinlaenge ────────────────────────────────────────────────────────────────

# Fallback falls weder leg_length_m noch body_height_m in subjects.csv stehen
# TODO: ersetzen sobald echte Daten vorliegen
LEG_LENGTH_FALLBACK_M = 1.0

# Schaetzfaktor Beinlaenge aus Koerpergroesse (De Leva, 1996)
# leg_length = body_height * LEG_LENGTH_FACTOR
LEG_LENGTH_FACTOR = 0.53


# ── Clustering ────────────────────────────────────────────────────────────────

# Mindestanzahl Probanden fuer die Analyse
# Wird nach Schritt 2 geprueft – bei Unterschreitung bricht das Programm ab
MIN_SUBJECTS = 3

# Zu testende Clusteranzahlen fuer k-Means und hierarchisches Clustering
# Bereich anpassen je nach erwarteter Gruppenanzahl
K_RANGE = range(2, 7)  # testet k = 2, 3, 4, 5, 6

# Welche Methoden sollen verglichen werden?
# True = aktiv, False = ueberspringen
RUN_KMEANS        = True
RUN_HIERARCHICAL  = True
RUN_HDBSCAN       = True

# Linkage-Methoden fuer hierarchisches Clustering
# Empfehlung fuer BA: ward + complete als Hauptmethoden
# single und average zum Vergleich
HIERARCHICAL_LINKAGES = ["ward", "complete", "average", "single"]

# HDBSCAN Parameter-Grid (nur relevant wenn RUN_HDBSCAN = True)
HDBSCAN_MIN_CLUSTER_SIZES = [3, 4, 5]
HDBSCAN_MIN_SAMPLES       = [None, 2, 3]

# Reproduzierbarkeit: fixer Zufallsseed fuer k-Means
RANDOM_STATE = 42


# ── Ausgabe ───────────────────────────────────────────────────────────────────

# Dateinamen fuer Ergebnisse
CLUSTER_RESULTS_CSV     = OUTPUT_DATA_DIR / "cluster_results.csv"
FATIGUE_FEATURES_CSV    = OUTPUT_DATA_DIR / "fatigue_features.csv"
CLUSTER_LABELS_CSV      = OUTPUT_DATA_DIR / "cluster_labels.csv"

# Plot-Einstellungen
PLOT_DPI    = 300   # Aufloesung fuer gespeicherte Plots
PLOT_FORMAT = "png" # "png" oder "pdf"


