"""Verify prereg_v2/out/*.npz against prereg_v2/out_manifest.csv (SHA-256 of each curve file).

Usage (from the repository root, after unpacking the curve archive into prereg_v2/out/):
    python prereg_v2/verify_manifest.py

Exit code 0 if all listed files exist and match, 1 otherwise. This checks byte identity with the archived main run;
a fresh run in another environment will generally NOT be byte-identical (see REPRODUCE.md, level 1 vs 2 vs 3).
"""

import csv
import hashlib
import sys
from pathlib import Path

OUT = Path("prereg_v2/out")
MANIFEST = Path("prereg_v2/out_manifest.csv")


def main() -> int:
    with MANIFEST.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    missing, bad = [], []
    for r in rows:
        f = OUT / r["file"]
        if not f.exists():
            missing.append(r["file"])
            continue
        if hashlib.sha256(f.read_bytes()).hexdigest() != r["sha256"]:
            bad.append(r["file"])
    print(f"manifest entries: {len(rows)}; missing: {len(missing)}; hash mismatches: {len(bad)}")
    for name in (missing + bad)[:10]:
        print("  ", name)
    return 0 if not missing and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
