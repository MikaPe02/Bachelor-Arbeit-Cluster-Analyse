"""
Untersuche .mat-Datei-Struktur um Speed-Feld zu finden
"""

from scipy.io import loadmat
import numpy as np

# Lade eine Beispiel-.mat-Datei
mat_path = r"C:\Users\Mika\Uni\BA\Beispiel Daten\P61_km01_0.mat"  # Pfad anpassen!
mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

# Inspiziere Struktur
print("TOP-LEVEL KEYS:")
print(mat.keys())
print("\n" + "="*70)

# PARAMETERS Struktur
if 'PARAMETERS' in mat:
    params = mat['PARAMETERS']
    print("\nPARAMETERS Felder:")
    if hasattr(params, '_fieldnames'):
        for field in params._fieldnames:
            print(f"  - PARAMETERS.{field}")
            
    if hasattr(params, 'R'):
        print("\nPARAMETERS.R Felder:")
        if hasattr(params.R, '_fieldnames'):
            for field in params.R._fieldnames:
                value = getattr(params.R, field)
                print(f"  - PARAMETERS.R.{field} = {value}")

# CONTACT Struktur
if 'CONTACT' in mat:
    contact = mat['CONTACT']
    print("\nCONTACT Felder:")
    if hasattr(contact, '_fieldnames'):
        for field in contact._fieldnames:
            print(f"  - CONTACT.{field}")
            trial = getattr(contact, field)
            if hasattr(trial, '_fieldnames'):
                for subfield in trial._fieldnames:
                    print(f"      - CONTACT.{field}.{subfield}")

# Suche nach "Speed", "Velocity", "Pace"
print("\n" + "="*70)
print("SUCHE NACH SPEED-VARIABLEN:")

def search_speed_recursively(obj, prefix=""):
    """Rekursive Suche nach Speed/Velocity"""
    if hasattr(obj, '_fieldnames'):
        for field in obj._fieldnames:
            if any(keyword in field.lower() for keyword in ['speed', 'velocity', 'pace']):
                value = getattr(obj, field)
                print(f"  ✅ GEFUNDEN: {prefix}.{field} = {value}")
            # Rekursiv weitersuchen
            search_speed_recursively(getattr(obj, field), f"{prefix}.{field}")
            
search_speed_recursively(mat, "ROOT")