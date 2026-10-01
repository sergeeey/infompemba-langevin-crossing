# legacy_v1 - material of the earlier (v1) draft

Status: superseded. The v1 claim of a "Mpemba-like crossing" is not supported (see `paper/note_v2_draft.md`, section 5, and
`prereg_v2/RESULTS.md`). These files are kept for provenance, not as results.

**What is here (moved 2026-10-01, content unchanged):** the original task statement (`TZ.txt`), the launch plan, the arXiv/Research Square
submission bundle and LaTeX build leftovers, and the preprint structure notes.

**What deliberately stays at the repository root**, because the v2 note, the checkers or the scripts themselves depend on the old paths:
- `run_phase_diagram.py` (the note cites lines 92-99 of it), `results/` (v1 phase map and data read by `prereg_v2/reproduce_v1_artifact.py`
  and `v1_*` scripts), `paper/preprint_v1.md`, `paper/preprint.tex`, `paper/preprint.pdf`, `paper/preprintNotes.bib`, the figures;
- the v1 GPU/ML experiment code (`run_*.py`, `launch_full_experiment.py`, `src/fisher_metrics.py`, `src/initializers.py`,
  `src/langevin_sgd.py`, `src/mpemba_runner.py`, `configs/`, `analysis/`, `requirements.txt`, `Dockerfile`): these scripts locate the project
  by their own directory and would stop running if moved. The ML experiments are classed in the note as invalid (K4) and are not needed to
  check the v2 re-analysis.

The frozen tag `prereg-v2-frozen` and the Zenodo records are unaffected by this move; they contain the earlier tree.
