# Bachelorarbeit – Laufermüdung via biomechanisches Clustering

Analyse von Laufermüdung anhand des **Dual-Axis Frameworks** nach van Oeveren et al. (2021).
Aus Motion-Capture-Daten (`.mat`-Dateien) werden Duty Factor (DF) und normierte
Schrittfrequenz (SF_norm) extrahiert und Probanden über unsupervised Clustering
nach Ermüdungsmustern gruppiert.

---

## Setup

```bash
conda activate fatigue
```

Abhängigkeiten: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`

Optional (dichtebasiertes Clustering):
```bash
pip install hdbscan
```
Fehlt `hdbscan`, läuft die Pipeline normal durch. Wird HDBSCAN gewählt oder `RUN_HDBSCAN = True` gesetzt, erscheint eine klare Fehlermeldung mit dem obigen Install-Befehl.

---

## Schnellstart mit synthetischen Testdaten

```bash
# 1. Synthetische Testdaten generieren (30 Probanden, 5 Gruppen)
python tests/generate_test_data.py

# 2. Analyse-Pipeline ausführen (inkl. interaktiver Methodenwahl + Plots)
python main.py
```

---

## Workflow mit echten Daten

```bash
# 1. DF und SF_norm aus MAT-Dateien extrahieren
#    -> Beim Start öffnet sich ein Ordner-Auswahldialog
python extract_to_csv.py

# 2. Analyse-Pipeline ausführen (inkl. interaktiver Methodenwahl + Plots)
python main.py
```

---

## Outputs

| Datei | Inhalt |
|---|---|
| `Outputs/Data/fatigue_features.csv` | Delta und Slope (DF, SF_norm) pro Proband |
| `Outputs/Data/cluster_results.csv` | Silhouette, Davies-Bouldin, CH-Score je Methode/k |
| `Outputs/Data/cluster_labels.csv` | Finale Cluster-Zuweisung pro Proband |
| `Outputs/Plots/elbow_plot.png` | Elbow-Methode: Innerhalb-Cluster-Streuung vs. k |
| `Outputs/Plots/dual_axis_snapshot.png` | Ausgangslaufstile bei km 1.0 |
| `Outputs/Plots/dual_axis_arrows.png` | Ermüdungsverlauf km 1.0 → 9.5 |
| `Outputs/Plots/cluster_scatter.png` | Clustering-Ergebnis mit Zentroiden |
| `Outputs/Plots/metrics_table.png` | Validierungsmetriken aller Methoden als Tabelle |
| `Outputs/Plots/dendrogram_plot.png` | Dendrogram (nur bei hierarchischem Clustering) |

---

## Projektstruktur

```
main.py                          # Haupt-Pipeline (interaktiv)
config.py                        # Zentrale Konfiguration
extract_to_csv.py                # MAT → dual_axis_dataset.csv
tests/generate_test_data.py      # Synthetische Testdaten
exploration/20_create_plots.py   # Plots standalone neu erzeugen

Bachelor-Arbeit-Cluster-Analyse/src/
  fatigue/
    io.py                        # MAT-Datei laden
    features.py                  # DF, SF, SF_norm berechnen
    fatigue_metrics.py           # Delta + Slope pro Proband
    preprocessing.py             # Z-Transformation, Sanity-Check
    clustering_eval.py           # k-Means, hierarchisch, HDBSCAN; Methoden-Vergleich
    clustering_ui.py             # Interaktive Terminal-Menüs zur Methodenwahl
  extension/
    viz_plots.py                 # Dual-Axis Visualisierungen + Elbow-Plot + Metriken-Tabelle

Input/
  subjects.csv                   # Körpergröße / Beinlänge pro Proband
  processed/dual_axis_dataset.csv
```

---

## Konfiguration (`config.py`)

| Parameter | Bedeutung |
|---|---|
| `DATA_RAW_FOLDER` | `None` – Ordner wird per Dialog in `extract_to_csv.py` gewählt |
| `MIN_SUBJECTS` | Mindestanzahl Probanden (Standard: 3) |
| `K_RANGE` | Zu testende Clusteranzahlen |
| `RUN_KMEANS / RUN_HIERARCHICAL / RUN_HDBSCAN` | Methoden ein-/ausschalten |
| `RANDOM_STATE` | Seed für Reproduzierbarkeit (42) |

---

## Dual-Axis Framework

SF_norm = SF × √(l₀/g), wobei l₀ die Beinlänge (Trochanter major → Boden) ist.
Beinlänge wird aus `Input/subjects.csv` gelesen. Falls nur Körpergröße bekannt:
l₀ ≈ 0.53 × Körpergröße (De Leva, 1996).

**Ermüdungsfeatures pro Proband:**
- **Delta** = letzter Wert − erster Wert (sensitiv für Endspurt)
- **Slope** = Steigung der linearen Regression über alle km-Marken (robust)

---

*Rohdaten (`.mat`, `.c3d`) und lokale Outputs sind nicht im Repository.*
