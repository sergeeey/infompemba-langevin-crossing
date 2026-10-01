# Request for independent human review

**Status of the work.** `paper/note_v2_draft.md` (draft v0.6, not for submission) is a correction note and numerical case study. It is
**not an independent replication**: the solver, the checks, four "skeptic" passes and the code reviews were all done with AI assistance
in one workflow, and no human other than the author has checked it. The aim of this request is to replace that gap with a real
check, including a negative one. A finding that the analysis is wrong is a useful result and will be recorded as such.

**What I am asking for.** Any one of the three tasks below is useful; they are listed from cheapest to most demanding. Please say
which ones you did. Nothing here needs the GPU/ML experiments of the earlier project.

## Task 1 - rerun the checks (about 30 minutes)

Follow `REPRODUCE.md` sections 1-3 on your own machine:

```bash
python -m pytest -q tests                           # expect: 45 passed
python prereg_v2/check_note_claims.py prereg_v2     # expect: 56 of 56 claims reproduced (needs the curve archive, REPRODUCE.md section 3)
python prereg_v2/verify_manifest.py                 # expect: all 288 SHA-256 values match
python prereg_v2/check_guard_on_main_run.py         # expect: 0 KL/W1 labels changed
```

Report your Python/NumPy/SciPy/Numba versions and anything that differs from the expected output. (`check_note_claims.py` only checks
that the note transcribes the pipeline's own outputs; it is not an independent derivation.)

## Task 2 - read the solver and the classifier (a few hours)

Read `src/mpemba_exact.py` (Fokker-Planck solver, distances, slow mode) and `src/mpemba_analysis.py` (`classify_metric`, P0, EFFECT).
Please look for: an error in the flux/boundary treatment, a wrong sign or definition in KL or W1, a way the classifier can return
EFFECT for a comparison that is not meaningful, and any place where the code does not do what Section 2-3 of the note says.
Line-level comments are the most useful form.

## Task 3 - recompute a few configurations independently (a day)

Write your own solver (any method; it need not be Scharfetter-Gummel) and compare with the stored numbers. Definitions, so that you do
not need to read my code:

- Model: x-marginal of an overdamped Langevin particle, dx = -U'(x) dt + sqrt(2T) dW, with U(x) = b (x^2 - 1)^2 + kappa b x
  (the y coordinate is an independent Ornstein-Uhlenbeck process and does not enter). Reflecting walls at |x| = 15.
- Equilibrium: pi proportional to exp(-U/T).
- Cold state: pi restricted to x > 0 and renormalised. Hot state: Gaussian N(0, 3^2) on the grid, renormalised.
- Distances to pi: KL(rho_t || pi) and Wasserstein-1 (W1) of the x marginal.
- P0 per metric: D_hot(0) / D_cold(0) >= 1.25, otherwise NO_TEST.
- EFFECT: gap = D_cold - D_hot > 5 x (summed error estimates) at every recorded time from some t* on, t* < t_max / 1.05,
  t_max = 8 / lambda_1, ignoring times where D_cold < 1e-18. You will not have my error estimates; use a convergence check of your
  own and say what margin you used.
- lambda_1 is the slowest relaxation rate; c1 = sum_i g1_i rho_i is the overlap with the slowest left eigenvector and r = |c1(hot)| / |c1(cold)|.

Reference numbers (from `prereg_v2/summary_main.csv`; regenerate with `python prereg_v2/list_review_configs.py`):

| b | T | kappa | lambda_1 | KL label | KL P0 ratio | W1 label | W1 P0 ratio | r |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.2 | 0 | 0.0111061 | EFFECT | 1626.18 | EFFECT | 1.64454 | 2.6e-12 (hot c1 is numerically zero by symmetry) |
| 2 | 0.2 | 0 | 0.000156762 | EFFECT | 3255.57 | EFFECT | 1.6241 | 1.0e-9 (hot c1 is numerically zero by symmetry) |
| 1 | 0.1 | 0.1 | 0.000116582 | EFFECT | 1080.04 | EFFECT | 1.28505 | 0.425296 |
| 0.1 | 1 | 0.1 | 0.62442 | EFFECT | 30.4557 | EFFECT | 1.35139 | 0.0150664 |
| 3 | 0.3 | 0.05 | 0.000262513 | EFFECT | 1738.72 | EFFECT | 1.34912 | 0.309666 |
| 1.5 | 0.05 | 0.1 | 2.3e-12 (first-passage estimate) | EFFECT | 1136.56 | **NO_TEST** | **1.24869** | not evaluated (b/T > 12) |

The last row can be checked from the initial distributions alone (no dynamics): the W1 precondition misses 1.25 by about 0.001.
Please report in particular any configuration where your label or your P0 ratio differs, and the value of lambda_1 you find.
Before reading the code, you may also try to predict the sign of the late-time KL gap at kappa = 0 from symmetry alone; that is the
design-determined part of the result.

## What would count as a negative result

Any of: a different label at several of the configurations above; a different lambda_1 or r; an initial-ordering or boundary error in
the solver; a P0 ratio that differs from the table; or a reading of the note that I should change. These are listed in the note's
section 6 ("What would change these conclusions").

## How to respond

Copy `reviews/TEMPLATE.md`, fill it in, and send it to me or open an issue/pull request. Responses, including negative ones, will be
recorded verbatim in `reviews/` with your name or a pseudonym as you prefer. Please state any conflict of interest, and whether you
used AI tools yourself. I will not edit a response; I will answer it separately.

## Suggested message (for the author to adapt and send; nothing is sent automatically)

> I have a small correction note with a reproducibility package on an earlier claim of mine of a "Mpemba-like crossing" in a double-well
> Langevin model (the effect turns out to be a metric artefact in my earlier analysis; the re-analysis shows the known spectral
> mechanism). Everything was done with AI assistance and has had no human check. Would you spend 30 minutes on task 1 or longer on
> task 2 or 3 of `REVIEW_REQUEST.md` in https://github.com/sergeeey/infompemba-langevin-crossing (branch `feature/prereg-v2`)? A
> negative finding is exactly what I am after.
