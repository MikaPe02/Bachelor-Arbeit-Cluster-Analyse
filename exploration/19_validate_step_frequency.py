# Skribte/19_validate_step_frequency.py
#
# ZWECK: Validierung der eigenen SF-Berechnung gegen den vorberechneten
#        StepFrequency-Wert aus der MAT-Datei.
#
# ERWARTETES ERGEBNIS: Abweichung < 1% → eigene Berechnung ist korrekt
#                      Abweichung > 5% → Problem, muss untersucht werden

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parents[1]
src_path = project_root / "Bachelor-Arbeit-Cluster-Analyse" / "src"
sys.path.append(str(src_path))

from fatigue.io import find_data_files, load_mat, extract_dual_axis_parameters

# ── Konfiguration ────────────────────────────────────────────────────────────
data_folder = r"C:\Users\Mika\Uni\BA\Beispiel Daten"  # ← ggf. anpassen

# TODO: Beinlänge pro Proband noch nicht bekannt – Platzhalter bis Daten vorliegen
# Quelle erfragen: Studienleiter kontaktieren
# Fallback-Schätzung falls nötig: l0 ≈ 0.53 × Körpergröße (De Leva, 1996)
LEG_LENGTH_PLACEHOLDER = 1.0
# ─────────────────────────────────────────────────────────────────────────────

files = find_data_files(data_folder)
mat_files = [f for f in files if f.suffix.lower() == ".mat"]

if not mat_files:
    raise SystemExit("Keine MAT-Dateien gefunden.")

print(f"Validierung auf {len(mat_files)} Datei(en)\n")
print(f"{'Datei':<30} {'SF_mat':>10} {'SF_calc':>10} {'Abw_%':>8} {'Status':>10}")
print("-" * 72)

all_ok = True

for mat_path in mat_files:
    mat = load_mat(mat_path)
    p = extract_dual_axis_parameters(mat)

    ct = np.asarray(p["ContactTimes"], dtype=float).flatten()
    ft = np.asarray(p["FlightTimes"], dtype=float).flatten()

    # Vorberechneter Wert aus MAT
    sf_mat = float(np.asarray(p["StepFrequency"]).flatten()[0])

    # Unsere eigene Berechnung
    n = min(len(ct), len(ft))
    stride_times = ct[:n] + ft[:n]
    sf_calc = float(np.mean(1.0 / stride_times[stride_times > 0]))

    # Abweichung
    abweichung_pct = abs(sf_mat - sf_calc) / sf_mat * 100

    status = "✓ OK" if abweichung_pct < 1.0 else ("⚠ PRÜFEN" if abweichung_pct < 5.0 else "✗ FEHLER")
    if abweichung_pct >= 1.0:
        all_ok = False

    print(f"{mat_path.name:<30} {sf_mat:>10.4f} {sf_calc:>10.4f} {abweichung_pct:>7.2f}% {status:>10}")

print("-" * 72)
if all_ok:
    print("\n✓ Alle Dateien: Abweichung < 1% → eigene Berechnung ist korrekt.")
else:
    print("\n⚠ Mindestens eine Datei hat Abweichung ≥ 1% → Ausgabe oben prüfen.")

print("\nHinweis: leg_length ist aktuell ein Platzhalter (1.0 m).")
print("TODO: Echte Beinlängen pro Proband einpflegen sobald Daten vorliegen.")