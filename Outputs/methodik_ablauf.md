# Methodischer Ablauf der Datenanalyse

---

## 0. Voraussetzungen / Rohdaten

- **Rohdaten**: `.mat`-Dateien aus dem Bewegungsanalyse-Labor (MATLAB-Export)
- **Dateinamenschema**: `{Proband}_km{INT}_{DEC}.mat` (z. B. `P61_km01_5.mat` = Proband P61, km 1.5)
- **Probandeninfo**: `Input/subjects.csv` mit Anthropometrie (Körpergröße, Gewicht, Beinlänge) und Laufgeschwindigkeit (`speed_ms`)
- N = 60 Probanden (Ausgangsstichprobe), je bis zu 10 km-Marken (km 1–10, ganzzahlig)
- **Ausschluss**: Probanden ohne km-10-Messung werden vollständig aus der Analyse ausgeschlossen (Begründung: Delta und Slope setzen vollständigen Verlauf km 1–10 voraus; unvollständige Daten würden Ermüdung systematisch unterschätzen)

---

## 1. Merkmalsextraktion aus den Rohdaten

**Skript**: `extract_to_csv.py`

Aus jeder `.mat`-Datei werden folgende Parameter extrahiert:

### Duty Factor (DF)
```
DF = Kontaktzeit / (Kontaktzeit + Flugzeit)
```
- Aus den Feldern `CONTACT.<trial>.ContactTimes` und `CONTACT.<trial>.FlightTimes`
- Mittelwert über alle Schritte pro km-Marke
- Dimensionslos [–], Wertebereich der Stichprobe: 0.491–0.850

### Schrittfrequenz normiert (SF_norm)
```
SF_norm = SF_Hz × sqrt(Beinlänge / g)
```
- SF in Hz wird aus Kontakt- und Flugzeiten berechnet (der im `.mat`-File gespeicherte Wert `PARAMETERS.R.StepFrequency` ist ungültig: INT32_MAX)
- Beinlänge: Priorität (1) gemessen aus `subjects.csv`, (2) geschätzt via De-Leva-Faktor 0.53 × Körpergröße
- g = 9.81 m/s²
- Dimensionslos [–], Wertebereich der Stichprobe: 0.746–0.972

**Output**: `Input/processed/dual_axis_dataset.csv`
Spalten: `Subject, km, DF, SF_norm, speed_ms`

---

## 2. Berechnung der Fatigue-Features

**Modul**: `src/fatigue/fatigue_metrics.py` → `build_fatigue_feature_table()`

Probanden ohne km-10-Messung werden hier identifiziert und **komplett aus dem Datensatz entfernt** — also auch aus dem Laufstil-Clustering (Schritt 5). Dies geschieht bevor jegliche weitere Analyse stattfindet.

Für jeden verbleibenden Probanden werden aus dem Verlauf km 1–10 vier Ermüdungsmerkmale berechnet:

| Feature    | Berechnung                                        |
|------------|---------------------------------------------------|
| Delta_DF   | DF(km 10) − DF(km 1)                             |
| Slope_DF   | Steigung der Regressionsgeraden DF über km 1–10   |
| Delta_SF   | SF_norm(km 10) − SF_norm(km 1)                   |
| Slope_SF   | Steigung der Regressionsgeraden SF_norm über km 1–10 |

Zusätzlich: `DF_start`, `DF_end`, `SF_start`, `SF_end` (km 1 und km 10 Werte)

Diese Features werden **nicht** als Input für das Laufstil-Clustering verwendet, sondern erst im Fatigue-Clustering (Schritt 6).

**Output**: `Outputs/Data/fatigue_features.csv`

---

## 3. Voranalyse: Speed-Korrelation

**Skript**: `exploration/01_speed_correlation_analysis.py`

Pearson-Korrelation zwischen Laufgeschwindigkeit und den Clustering-Merkmalen bei km 1:

| Variable | r      | R²    | p       |
|----------|--------|-------|---------|
| DF       | −0.749 | 0.561 | < .001  |
| SF_norm  | +0.454 | 0.206 | < .001  |

**Entscheidung**: Trotz starker Korrelation (r = −.75 für DF) wird auf eine Speed-Bereinigung verzichtet. Speed wird als inhärentes Merkmal des Laufstils betrachtet — wer schneller läuft, hat strukturell einen niedrigeren Duty Factor. Eine Residualbereinigung würde diesen inhaltlichen Zusammenhang herausrechnen und den Laufstil verfälschen.

---

## 4. Stichprobenbeschreibung

**Skript**: `exploration/03_descriptive_stats.py`

Deskriptive Statistik der Analysestichprobe nach Ausschluss von 8 Probanden ohne km-10-Messung (N = 52):

| Variable           |   MW  |    SD |   Min |   Max |
|--------------------|------:|------:|------:|------:|
| Körpergröße [m]    | 1.772 | 0.090 | 1.570 | 1.940 |
| Gewicht [kg]       | 69.88 | 10.68 | 46.70 | 91.90 |
| Beinlänge [m]      | 0.908 | 0.054 | 0.772 | 1.000 |
| Speed [m/s]        | 3.554 | 0.619 | 2.646 | 5.190 |
| SF bei km 1 [Hz]   | 2.860 | 0.147 | 2.616 | 3.361 |
| Duty Factor (km 1) | 0.690 | 0.077 | 0.491 | 0.850 |
| SF_norm (km 1)     | 0.871 | 0.037 | 0.746 | 0.972 |

---

## 5. Laufstil-Clustering (Ward, k = 4)

**Skript**: `main.py`  
**Modul**: `src/fatigue/clustering_eval.py`

### 5.1 Clustering-Input

- **Features**: DF und SF_norm bei km 1.0 (Rohdaten, keine Speed-Bereinigung)
- **Standardisierung**: z-Transformation (StandardScaler, µ = 0, σ = 1) separat für DF und SF_norm

### 5.2 Clustering-Verfahren

- **Methode**: Ward-Hierarchisches Clustering (`scipy.cluster.hierarchy`, Ward-Linkage)
- **Distanzmaß**: Euklidische Distanz im z-transformierten Merkmalsraum
- **Ward-Kriterium**: Minimierung der Gesamtstreuung innerhalb der Cluster (Within-Cluster Sum of Squares)
- **Clusteranzahl**: k = 4 (manuell gewählt auf Basis des Dendrogramms und der Gütekriterien)

### 5.3 Gütekriterien (Ward-Clustering, alle k) — N = 52

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|-----------|---------------|------------------|
| 2 |     0.303 |         0.981 |             20.2 |
| 3 |     0.351 |         0.962 |             28.2 |
| **4** | **0.382** | **0.942** | **32.9** | ← gewählt |
| 5 |     0.391 |         0.760 |             35.1 |
| 6 |     0.341 |         0.696 |             36.8 |
| 7 |     0.363 |         0.753 |             37.0 |
| 8 |     0.380 |         0.716 |             39.6 |

### 5.4 Cluster-Größen (Stil) — N = 52

| Cluster   | N  |
|-----------|-----|
| Cluster 1 | 10 |
| Cluster 2 | 12 |
| Cluster 3 | 22 |
| Cluster 4 |  8 |

### 5.5 Silhouette-Koeffizient pro Cluster

| Cluster   |  N | MW    |  SD   |    Min |   Max | Grenzfälle (s < 0.2) |
|-----------|----|------:|------:|-------:|------:|---------------------:|
| Cluster 1 | 10 | 0.139 | 0.211 | −0.234 | 0.441 |  7 |
| Cluster 2 | 12 | 0.475 | 0.113 |  0.308 | 0.645 |  0 |
| Cluster 3 | 22 | 0.428 | 0.116 |  0.196 | 0.602 |  1 |
| Cluster 4 |  8 | 0.417 | 0.161 |  0.184 | 0.590 |  1 |
| **Gesamt**| 52 | **0.382** | 0.186 | −0.234 | 0.645 | **9** |

### 5.6 Gespeicherte Outputs

```
Outputs/Data/cluster_labels.csv              — Cluster-Zuordnung pro Proband
Outputs/Data/cluster_results.csv             — Gütekriterien je k
Outputs/Plots/dendrogram_hierarchisch_ward_k4_roh.png
Outputs/Plots/cluster_scatter_hierarchisch_ward_k4_roh.png
Outputs/Plots/dual_axis_arrows_hierarchisch_ward_k4_roh.png
Outputs/Plots/dual_axis_snapshot.png
Outputs/Plots/elbow_plot_hierarchisch_ward_k4_roh.png
```

---

## 6. Deskriptive Statistik und Inferenzstatistik — Laufstil-Cluster

**Modul**: `src/extension/descriptive.py` → `describe_clusters()`

### 6.1 Deskriptive Statistik (MW ± SD pro Cluster)

| Variable          | Cluster 1      | Cluster 2      | Cluster 3      | Cluster 4      |
|-------------------|---------------|---------------|---------------|---------------|
| DF [–]            | 0.682 ± 0.054 | 0.590 ± 0.035 | 0.767 ± 0.036 | 0.713 ± 0.032 |
| SF_norm [–]       | 0.836 ± 0.036 | 0.888 ± 0.024 | 0.867 ± 0.016 | 0.923 ± 0.025 |
| Speed [m/s]       | 3.308 ± 0.427 | 4.368 ± 0.424 | 3.128 ± 0.361 | 3.526 ± 0.225 |
| Körpergröße [m]   | 1.753 ± 0.071 | 1.783 ± 0.086 | 1.780 ± 0.112 | 1.788 ± 0.102 |
| Gewicht [kg]      | 68.6 ± 9.8    | 66.7 ± 10.5   | 73.6 ± 11.1   | 71.9 ± 12.0   |
| Beinlänge [m]     | 0.897 ± 0.053 | 0.910 ± 0.053 | 0.919 ± 0.055 | 0.911 ± 0.057 |
| Delta_DF [–]      | 0.003 ± 0.016 | 0.006 ± 0.013 | −0.003 ± 0.017| −0.007 ± 0.021|
| Slope_DF [–/km]   | 0.000 ± 0.002 | 0.001 ± 0.002 | 0.000 ± 0.002 | −0.001 ± 0.002|
| Delta_SF [–]      | −0.010 ± 0.015| −0.005 ± 0.016| −0.018 ± 0.020| −0.024 ± 0.024|
| Slope_SF [–/km]   | −0.001 ± 0.002| −0.000 ± 0.002| −0.002 ± 0.002| −0.003 ± 0.002|

### 6.2 Voraussetzungsprüfung

- **Normalverteilung**: Shapiro-Wilk-Test pro Variable und Cluster
- **Varianzhomogenität**: Levene-Test pro Variable
- **Testmethode**: einstufige ANOVA (alle Variablen einheitlich, interaktiv gewählt)

### 6.3 ANOVA-Ergebnisse Laufstil-Cluster

| Kategorie   | Variable        |      F |      p |   η²  | sig  |
|-------------|-----------------|-------:|-------:|------:|------|
| Biomechanik | DF              | 45.837 | < .001 | 0.711 | ***  |
| Biomechanik | SF_norm         | 22.863 | < .001 | 0.550 | ***  |
| Biomechanik | speed_ms        | 31.155 | < .001 | 0.625 | ***  |
| Biomechanik | body_height_m   |  0.525 |  .667  | 0.027 | n.s. |
| Biomechanik | weight_kg       |  1.297 |  .285  | 0.065 | n.s. |
| Biomechanik | leg_length_m    |  0.548 |  .651  | 0.029 | n.s. |
| Fatigue     | DF_start        | 45.837 | < .001 | 0.711 | ***  |
| Fatigue     | DF_end          | 36.071 | < .001 | 0.659 | ***  |
| Fatigue     | Delta_DF        |  1.481 |  .230  | 0.073 | n.s. |
| Fatigue     | Slope_DF        |  1.897 |  .141  | 0.092 | n.s. |
| Fatigue     | Slope_DF_late   |  5.304 |  .003  | 0.221 | **   |
| Fatigue     | SF_start        | 22.863 | < .001 | 0.550 | ***  |
| Fatigue     | SF_end          | 13.202 | < .001 | 0.414 | ***  |
| Fatigue     | Delta_SF        |  2.444 |  .073  | 0.116 | n.s. |
| Fatigue     | Slope_SF        |  3.316 |  .026  | 0.151 | *    |

Post-hoc: Tukey-HSD (nur bei signifikanter ANOVA). Effektgröße: Cohen's d pro Clusterpaar.

### 6.4 Chi-Quadrat (kategoriale Variablen)

| Variable     | χ²    | df | p     | sig  |
|--------------|------:|----|------:|------|
| Geschlecht   | 2.345 |  3 | .504  | n.s. |
| Standbein    | 1.141 |  3 | .767  | n.s. |

---

## 7. Fatigue-Clustering (Ward, k = 3)

**Skript**: `main.py` → `step_fatigue_clustering()`  
**Modul**: `src/fatigue/clustering_eval.py`

### 7.1 Clustering-Input

- **Features**: Delta_DF, Slope_DF, Delta_SF, Slope_SF (aus Schritt 2)
- **Standardisierung**: z-Transformation (StandardScaler)
- N = 52 Probanden (nach Ausschluss der 8 Probanden ohne km-10-Messung)

### 7.2 Clustering-Verfahren

Identisch zu Schritt 5: Ward-Hierarchisches Clustering, Euklidische Distanz, k = 3.

### 7.3 Gütekriterien (Ward-Clustering, alle k) — N = 52

| k | Silhouette | Davies-Bouldin | Calinski-Harabasz |
|---|-----------|---------------|------------------|
| 2 |     0.395 |         0.898 |             37.2 |
| **3** | **0.398** | **0.901** | **40.7** | ← gewählt |
| 4 |     0.393 |         0.820 |             39.6 |
| 5 |     0.345 |         0.756 |             41.7 |
| 6 |     0.358 |         0.813 |             42.0 |
| 7 |     0.356 |         0.833 |             42.1 |
| 8 |     0.358 |         0.837 |             42.6 |

### 7.4 Cluster-Größen (Fatigue) — N = 52

| Cluster   |  N |
|-----------|----|
| Cluster 1 | 15 |
| Cluster 2 | 29 |
| Cluster 3 |  8 |

### 7.5 Silhouette-Koeffizient pro Cluster

| Cluster   |  N | MW    |  SD   |    Min |   Max | Grenzfälle (s < 0.2) |
|-----------|----|------:|------:|-------:|------:|---------------------:|
| Cluster 1 | 15 | 0.306 | 0.193 | −0.153 | 0.478 |  4 |
| Cluster 2 | 29 | 0.474 | 0.107 |  0.270 | 0.624 |  0 |
| Cluster 3 |  8 | 0.297 | 0.103 |  0.160 | 0.434 |  2 |
| **Gesamt**| 52 | **0.398** | 0.159 | −0.153 | 0.624 | **6** |

### 7.6 Gespeicherte Outputs

```
Outputs/Data/fatigue_cluster_labels.csv
Outputs/Data/fatigue_cluster_results.csv
Outputs/Plots/Fatigue/dendrogram_hierarchisch_ward_k3_roh.png
Outputs/Plots/Fatigue/fatigue_cluster_scatter_hierarchisch_ward_k3_roh.png
Outputs/Plots/Fatigue/elbow_plot_hierarchisch_ward_k3_roh.png
```

---

## 8. Deskriptive Statistik und Inferenzstatistik — Fatigue-Cluster

**Modul**: `src/extension/descriptive.py` → `describe_fatigue_clusters()`

### 8.1 Deskriptive Statistik (MW ± SD pro Cluster) — N = 52

**Block 1 — Ausgangsbedingungen (km 1)**

| Variable        | Cluster 1      | Cluster 2      | Cluster 3      |
|-----------------|---------------|---------------|---------------|
| DF_start [–]    | 0.666 ± 0.062 | 0.695 ± 0.088 | 0.719 ± 0.049 |
| SF_norm_start [–]| 0.867 ± 0.036 | 0.872 ± 0.033 | 0.874 ± 0.053 |
| Speed [m/s]     | 3.639 ± 0.631 | 3.519 ± 0.602 | 3.262 ± 0.351 |
| Körpergröße [m] | 1.747 ± 0.098 | 1.804 ± 0.082 | 1.747 ± 0.098 |
| Beinlänge [m]   | 0.895 ± 0.059 | 0.928 ± 0.043 | 0.900 ± 0.055 |

**Block 2 — Ermüdungsmerkmale (Clustering-Input)**

| Variable          | Cluster 1       | Cluster 2       | Cluster 3       |
|-------------------|----------------|----------------|----------------|
| Delta_DF [–]      |  0.014 ± 0.014 | −0.001 ± 0.008 | −0.030 ± 0.009 |
| Slope_DF [–/km]   |  0.001 ± 0.002 |  0.000 ± 0.001 | −0.003 ± 0.001 |
| Slope_DF_early    |  0.003 ± 0.003 |  0.001 ± 0.002 | −0.003 ± 0.003 |
| Slope_DF_late     |  0.001 ± 0.002 | −0.001 ± 0.002 | −0.004 ± 0.003 |
| Delta_SF [–]      |  0.009 ± 0.011 | −0.019 ± 0.011 | −0.035 ± 0.019 |
| Slope_SF [–/km]   |  0.001 ± 0.001 | −0.002 ± 0.001 | −0.004 ± 0.002 |
| Slope_SF_early    |  0.000 ± 0.002 | −0.002 ± 0.002 | −0.005 ± 0.005 |
| Slope_SF_late     |  0.001 ± 0.002 | −0.002 ± 0.002 | −0.003 ± 0.003 |

### 8.2 ANOVA-Ergebnisse Fatigue-Cluster — N = 52

| Kategorie   | Variable       |       F |      p |   η²  | sig  |
|-------------|----------------|--------:|-------:|------:|------|
| Biomechanik | DF             |   1.350 |  .269  | 0.052 | n.s. |
| Biomechanik | SF_norm        |   0.109 |  .897  | 0.004 | n.s. |
| Biomechanik | speed_ms       |   1.095 |  .343  | 0.043 | n.s. |
| Biomechanik | body_height_m  |   2.665 |  .080  | 0.098 | n.s. |
| Biomechanik | leg_length_m   |   2.567 |  .087  | 0.095 | n.s. |
| Fatigue     | Delta_DF       |  50.920 | < .001 | 0.675 | ***  |
| Fatigue     | Slope_DF       |  27.502 | < .001 | 0.529 | ***  |
| Fatigue     | Slope_DF_early |  15.100 | < .001 | 0.381 | ***  |
| Fatigue     | Slope_DF_late  |  18.158 | < .001 | 0.426 | ***  |
| Fatigue     | Delta_SF       |  40.089 | < .001 | 0.621 | ***  |
| Fatigue     | Slope_SF       |  50.153 | < .001 | 0.672 | ***  |
| Fatigue     | Slope_SF_early |   9.154 | < .001 | 0.272 | ***  |
| Fatigue     | Slope_SF_late  |  14.070 | < .001 | 0.365 | ***  |

### 8.3 Cohen's d (Cluster-Paare, Fatigue-Features)

| Variable       | C1 vs C2 | C1 vs C3 | C2 vs C3 |
|----------------|--------:|--------:|--------:|
| Delta_DF       |    1.467 |   3.525 |   3.753 |
| Slope_DF       |    1.065 |   2.557 |   2.748 |
| Slope_DF_early |    0.921 |   2.204 |   1.558 |
| Slope_DF_late  |    1.148 |   2.212 |   1.601 |
| Delta_SF       |    2.648 |   3.017 |   1.184 |
| Slope_SF       |    2.761 |   3.846 |   1.288 |
| Slope_SF_early |    1.163 |   1.535 |   0.790 |
| Slope_SF_late  |    1.530 |   1.706 |   0.667 |

### 8.4 Chi-Quadrat (kategoriale Variablen)

| Variable   |   χ²  | df |    p  | sig  |
|------------|------:|----|------:|------|
| Geschlecht | 0.601 |  2 | .740  | n.s. |
| Standbein  | 7.104 |  2 | .029  | *    |

---

## 9. Vergleich Laufstil- vs. Fatigue-Cluster

**Modul**: `src/extension/descriptive.py` → `compare_style_and_fatigue_clusters()`

Kreuztabelle (Zellen = Anzahl Probanden), N = 52:

|               | Fatigue C1 | Fatigue C2 | Fatigue C3 |
|---------------|----------:|----------:|----------:|
| **Stil C1**   |         4 |         5 |         1 |
| **Stil C2**   |         5 |         7 |         0 |
| **Stil C3**   |         5 |        13 |         4 |
| **Stil C4**   |         1 |         4 |         3 |

Chi-Quadrat-Test: χ²(6) = 6.996, p = .321 → kein signifikanter Zusammenhang zwischen Laufstil- und Fatigue-Clusterzugehörigkeit.

---

## 10. Verlaufsplots und Silhouette-Visualisierung

**Skript**: `exploration/22_verlaufsplot.py`  
Mittelwertverlauf von DF und SF_norm über km 1–10, getrennt für Laufstil- und Fatigue-Cluster.

**Skript**: `exploration/23_silhouette_plot.py`  
Balkendiagramm MW ± SD des Silhouette-Koeffizienten pro Cluster, getrennt für beide Clusterings.

---

## 11. APA-Tabellen-Export

**Skript**: `exploration/21_export_apa_tables.py`

Sechs CSV-Dateien im APA-Format (UTF-8-BOM für Google Sheets), jeweils für Laufstil und Fatigue:

| Datei                                     | Inhalt                                |
|-------------------------------------------|---------------------------------------|
| `apa_tabelle_deskriptiv_stil_*.csv`       | MW (SD) pro Cluster, alle Variablen   |
| `apa_tabelle_anova_stil_*.csv`            | F, p, η², Tukey-Zusammenfassung       |
| `apa_tabelle_metriken_stil_*.csv`         | Clustering-Gütekriterien k = 3/4/5    |
| `apa_tabelle_deskriptiv_fatigue_*.csv`    | MW (SD) pro Cluster, alle Variablen   |
| `apa_tabelle_anova_fatigue_*.csv`         | F, p, η², Tukey-Zusammenfassung       |
| `apa_tabelle_metriken_fatigue.csv`        | Clustering-Gütekriterien k = 2/3/4    |

---

## Software und Pakete

| Paket          | Verwendung                                      |
|----------------|-------------------------------------------------|
| Python 3.11    | Gesamte Analyse                                 |
| pandas         | Datenverwaltung, CSV-I/O                        |
| numpy          | Numerische Berechnungen                         |
| scipy          | Ward-Clustering (`hierarchy.linkage/fcluster`)  |
| scikit-learn   | StandardScaler, Silhouette, Davies-Bouldin, CH  |
| statsmodels    | Tukey-HSD-Post-hoc-Test                         |
| matplotlib     | Alle Plots (APA-Stil, 300 DPI)                  |
