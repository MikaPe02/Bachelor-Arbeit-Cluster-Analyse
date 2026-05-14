"""
Reorganisiert MAT-Dateien von einer flachen Struktur in Probanden-Ordner.

IST:  Input/raw/{ID}_km{N}_0.mat (alle in einem Ordner)
SOLL: Input/by_subject/{ID}/{ID}_km{N}_0.mat

ID-Extraktion: alles vor dem ersten "_km" im Dateinamen.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="MAT-Dateien nach Probanden-ID sortieren")
    p.add_argument("--source-dir", default="data/Raw_Data", help="Quellverzeichnis (default: Input/Raw_Data)")
    p.add_argument("--target-dir", default="data/Data", help="Zielverzeichnis (default: Input/Data)")
    p.add_argument("--mode", choices=["copy", "move"], default="copy", help="copy oder move (default: copy)")
    p.add_argument("--dry-run", action="store_true", help="Nur anzeigen, nichts ausführen")
    return p.parse_args()


def extract_subject_id(filename: str) -> str | None:
    """Extrahiert die Subject-ID via split auf '_km'."""
    if "_km" not in filename:
        return None
    return filename.split("_km")[0]


def extract_km_value(filename: str) -> int | None:
    """Extrahiert den km-Wert aus dem Dateinamen, z.B. 'P18_km10_0.mat' → 10."""
    try:
        after_km = filename.split("_km")[1]       # "10_0.mat"
        km_str = after_km.split("_")[0]            # "10"
        return int(km_str)
    except (IndexError, ValueError):
        return None


def build_progress_bar(done: int, total: int, width: int = 28) -> str:
    filled = int(width * done / total) if total else width
    if done < total:
        bar = "=" * filled + ">" + " " * (width - filled - 1)
    else:
        bar = "=" * width
    return f"[{bar}]"


def main():
    args = parse_args()

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    mode = args.mode
    dry_run = args.dry_run

    sep = "=" * 70
    print(sep)
    print("DATEN REORGANISATION: FLAT → PROBANDEN-ORDNER")
    print(sep)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Mode:   {mode}" + (" (DRY-RUN)" if dry_run else ""))
    print()

    # ── 1. Scan ──────────────────────────────────────────────────────────────
    print("Scanne Dateien...")
    if not source_dir.exists():
        print(f"  ❌ Quellverzeichnis nicht gefunden: {source_dir}", file=sys.stderr)
        sys.exit(1)

    mat_files = sorted(source_dir.rglob("*.mat"))
    print(f"  ✅ {len(mat_files)} .mat Dateien gefunden")
    print()

    # ── 2. ID-Extraktion ─────────────────────────────────────────────────────
    print("Extrahiere Subject-IDs via Pattern: {ID}_km{N}_0.mat")
    skipped: list[str] = []
    groups: dict[str, list[Path]] = {}

    for f in mat_files:
        subject_id = extract_subject_id(f.name)
        if subject_id is None:
            skipped.append(f.name)
            print(f"  ⚠️  Kein '_km' in Dateiname, überspringe: {f.name}", file=sys.stderr)
            continue
        groups.setdefault(subject_id, []).append(f)

    print(f"  ✅ {len(groups)} eindeutige IDs identifiziert")
    print()

    # ── 3. Vollständigkeit pro ID ─────────────────────────────────────────────
    ALL_KM = set(range(1, 11))
    completeness: dict[str, dict] = {}

    for subject_id, files in sorted(groups.items()):
        km_found = set()
        for f in files:
            km = extract_km_value(f.name)
            if km is not None:
                km_found.add(km)
        missing = sorted(ALL_KM - km_found)
        completeness[subject_id] = {
            "files": files,
            "n_files": len(files),
            "km_found": sorted(km_found),
            "missing_km": missing,
            "is_complete": len(missing) == 0,
        }

    # ── 4. ID-Beispiele ───────────────────────────────────────────────────────
    print("ID-Beispiele:")
    preview_ids = list(sorted(completeness.keys()))[:6]
    for sid in preview_ids:
        c = completeness[sid]
        km_str = ",".join(str(k) for k in c["km_found"])
        if c["is_complete"]:
            print(f"  {sid} ({c['n_files']} Dateien: km {km_str}) ✅")
        else:
            missing_str = ",".join(str(k) for k in c["missing_km"])
            print(f"  {sid} ({c['n_files']} Dateien: km {km_str}, fehlt: {missing_str}) ⚠️")
    if len(completeness) > 6:
        print(f"  ... ({len(completeness) - 6} weitere IDs)")
    print()

    # ── 5. Ordner anlegen ────────────────────────────────────────────────────
    print("Erstelle Ordner-Struktur...")
    for sid in sorted(completeness.keys()):
        folder = target_dir / sid
        print(f"  {'(dry-run) ' if dry_run else ''}✅ {folder}/")
        if not dry_run:
            folder.mkdir(parents=True, exist_ok=True)
    print()

    # ── 6. Dateien kopieren/verschieben ───────────────────────────────────────
    op_label = "Kopiere" if mode == "copy" else "Verschiebe"
    print(f"{op_label} Dateien...")
    total_transferred = 0
    max_id_len = max(len(s) for s in completeness) if completeness else 0

    for sid in sorted(completeness.keys()):
        c = completeness[sid]
        files = c["files"]
        n = len(files)
        bar = build_progress_bar(n, 10)
        status = "✅" if c["is_complete"] else "⚠️"
        padding = " " * (max_id_len - len(sid))
        print(f"  {sid}:{padding} {bar} {n}/10 {status}")

        if not dry_run:
            dest_folder = target_dir / sid
            for src in files:
                dest = dest_folder / src.name
                if mode == "copy":
                    shutil.copy2(src, dest)
                else:
                    shutil.move(str(src), dest)
                total_transferred += 1
        else:
            total_transferred += n

    print()

    # ── 7. Completeness-Report ────────────────────────────────────────────────
    report_path = target_dir / "completeness_report.csv"
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Subject_ID", "n_files", "km_found", "missing_km", "is_complete"])
            for sid in sorted(completeness.keys()):
                c = completeness[sid]
                writer.writerow([
                    sid,
                    c["n_files"],
                    ",".join(str(k) for k in c["km_found"]),
                    ",".join(str(k) for k in c["missing_km"]),
                    c["is_complete"],
                ])

    # ── 8. Zusammenfassung ────────────────────────────────────────────────────
    n_total = len(completeness)
    n_complete = sum(1 for c in completeness.values() if c["is_complete"])
    n_incomplete = n_total - n_complete
    pct = (n_complete / n_total * 100) if n_total else 0.0

    print(sep)
    print("ZUSAMMENFASSUNG")
    print(sep)
    print(f"IDs total:           {n_total}")
    print(f"Vollständig (10 km): {n_complete} ({pct:.1f}%)")
    print(f"Unvollständig:       {n_incomplete} ({100 - pct:.1f}%)")

    if n_incomplete:
        print()
        print("⚠️  UNVOLLSTÄNDIGE LÄUFE (nur Info, kein Fehler):")
        for sid in sorted(completeness.keys()):
            c = completeness[sid]
            if not c["is_complete"]:
                missing_str = ", ".join(str(k) for k in c["missing_km"])
                print(f"  {sid}: fehlt km {missing_str}")

    if skipped:
        print()
        print(f"⚠️  {len(skipped)} Dateien ohne '_km' im Namen übersprungen:")
        for name in skipped:
            print(f"  {name}")

    print()
    print(f"Dateien {'(dry-run) ' if dry_run else ''}{op_label.lower()}t: {total_transferred}")
    print(f"Zielordner:          {n_total}")
    print()
    if not dry_run:
        print(f"✅ Report: {report_path}")
    print(f"✅ FERTIG! Unvollständige Läufe wurden NICHT ausgeschlossen.")
    print(sep)
    print()
    print("HINWEIS FÜR ANALYSE:")
    print("Entscheiden Sie später ob unvollständige Läufe verwendet werden.")
    print("Für Clustering/Mixed Model empfohlen: Nur vollständige Läufe (10 km).")


if __name__ == "__main__":
    main()
