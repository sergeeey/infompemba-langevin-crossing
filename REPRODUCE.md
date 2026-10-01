# How to check or reproduce the v2 re-analysis

This is for someone who did not write the code. Everything here concerns the v2 re-analysis
(`paper/note_v2_draft.md`); the earlier GPU/ML experiments are not covered and are not needed.

## 0. What you can and cannot expect

Three different things can be "reproduced", and they need different amounts of trust:

| Level | What | Expected |
|---|---|---|
| 1. Byte identity | the 288 archived curve files match the manifest | yes, by checksum (section 3) |
| 2. Numerical agreement | a fresh run gives the same numbers to ~1e-9 relative | **not guaranteed across environments**: the strict `prereg_v2/check_equivalence.py` (rtol 1e-9) failed in an outside reviewer's environment (NumPy/SciPy/Numba versions differ), mostly near zero and in some W1 values |
| 3. Label stability | the scientific labels (EFFECT / NO_TEST / ...) agree | yes in the four configurations an outside reviewer recomputed; not checked for all 288 |

Nobody other than the author's own workflow has reproduced the whole main run. No human has replicated it.

## 1. Environment

```bash
git clone https://github.com/sergeeey/infompemba-langevin-crossing
cd infompemba-langevin-crossing
git checkout feature/prereg-v2          # or the release tag of the version you are checking
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements-v2.txt      # exact versions used for the stored results: requirements-v2.lock.txt
```

Do not use `requirements.txt` (that file is for the earlier GPU experiments; it lacks numba and pytest).

## 2. Fast checks (minutes)

```bash
python -m pytest -q tests                          # 45 tests (with the curve archive unpacked; 4 stored-curve tests are skipped without it)
python run_gates_v2.py                             # solver validation V1-V3 (about a minute)
python prereg_v2/counterexamples.py                # two small chains that bound what the spectral criterion R can claim
python prereg_v2/check_note_claims.py prereg_v2    # 56 numeric statements of the note against the CSV tables and curves
```

`check_note_claims.py` is a transcription-consistency check against the pipeline's own outputs, not an independent
re-derivation (see its docstring). It also reads the curve files for the margin statistics, so fetch the curve archive
first (section 3); without it the margin block fails.

## 3. The stored curves (Level 1)

The 288 per-configuration curves (`prereg_v2/out/*.npz`, about 42 MB) are not in the git tree. They are deposited
as a Zenodo dataset, DOI 10.5281/zenodo.23077448 (all versions: 10.5281/zenodo.23077447), split into six zip parts because of an upload-size limit.
Download all parts, unpack every part into `prereg_v2/` (each contains paths starting with `out/`), e.g.
`for f in infompemba_prereg_v2_curves_part*.zip; do unzip -o "$f" -d prereg_v2/; done`, and run:

```bash
python prereg_v2/verify_manifest.py     # SHA-256 of all 288 files against prereg_v2/out_manifest.csv
```

## 4. Re-running the main analysis (Levels 2-3)

```bash
python run_prereg_v2_main.py --smoke    # a few configurations, timing and crash check only
python run_prereg_v2_main.py            # 288 configurations; 386 s on 24 cores; uses (cores - 1) processes
python run_prereg_v2_main.py --summarize   # re-run the K-table logic on the saved CSV without recomputation
```

Compare the regenerated `prereg_v2/summary_main.csv` with the committed one **by labels first**, then by numbers with
an absolute tolerance near zero (do not require relative 1e-9 near exact zeros). Other scripts and what they produce are
listed in `prereg_v2/RESULTS.md` (header) and `prereg_v2/EXECUTION_LOG.md`.

## 5. What to distrust

- The pre-specification was written by the same AI-assisted workflow that ran the analysis; git dates are editable.
- The error estimates are estimates, not bounds; deep wells (b/T > 12) use a first-passage estimate of the slowest rate.
- The four skeptic passes and the code reviews were done by AI agents; no human review exists yet.
- The public git history was rewritten once on 2026-09-29 (see `HISTORY_REWRITE.md`); the frozen pre-specification
  `PREREG_v2.md` is unchanged (`PREREG_v2.sha256`, checked against the tagged content).
