"""D2.9c: is 'saved CSV + revised summarizer' equivalent to a fresh recomputation with the CURRENT code?

Recomputes 3 configurations with the current src/mpemba_analysis.py (which gained a t[0] == 0 check after
the main run) into a temporary directory and compares every field with the row in summary_main.csv:
labels must be equal, numbers equal to relative 1e-9. Read-only for the project outputs.
"""

import csv
import math
import sys
import tempfile

sys.path.insert(0, ".")
import run_prereg_v2_main as rm

CONFIGS = [(2.0, 0.2, 0.0), (5.0, 0.05, 0.02), (1.0, 0.1, 0.1)]

with open(rm.SUMMARY, encoding="utf-8") as fh:
    saved = {(float(r["b"]), float(r["T"]), float(r["kappa"])): r for r in csv.DictReader(fh)}

ok = True
with tempfile.TemporaryDirectory() as tmp:
    rm.OUT_DIR = tmp  # do not touch the committed manifest inputs
    for cfg in CONFIGS:
        fresh = rm.run_config(cfg)
        old = saved[cfg]
        bad = []
        for key, value in fresh.items():
            if key == "seconds":
                continue
            ref = old[key]
            if isinstance(value, str):
                same = value == ref
            else:
                a, b = float(value), float(ref)
                same = (
                    (math.isnan(a) and math.isnan(b))
                    or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-300)
                    or (math.isinf(a) and math.isinf(b))
                )
            if not same:
                bad.append((key, value, ref))
        ok &= not bad
        print(
            f"  {cfg}: {'IDENTICAL (labels equal, numbers within 1e-9)' if not bad else 'DIFFERENT'}"
            f" over {len(fresh) - 1} fields"
        )
        for item in bad[:5]:
            print("     ", item)
print(f"\nequivalence: {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
