"""Input guards for `classify_metric` (added 2026-10-01; NOT part of the pre-specification and NOT used for the main run).

`classify_metric` in `src/mpemba_analysis.py` trusts its input. In particular P0 is a non-strict comparison, so
D_hot(0) = D_cold(0) = 0 passes it with `p0_ratio = inf` and can end up labelled EFFECT, although two states that both
sit at equilibrium cannot show any Mpemba-type crossing. This module adds the missing checks WITHOUT editing
`classify_metric` (whose SHA-256 is quoted in the results files): it wraps it and returns the label INVALID_INPUT
when the input is not a meaningful comparison.

What is checked (every item is a statement about the input, not about the physics):
  - shapes: t is 1-D, d and err have shape (len(t), 2); all values finite; err >= 0;
  - t[0] == 0 and t strictly increasing;
  - the distances are non-negative up to their own error estimate (Richardson extrapolation can give small negative
    values; a value below -err is an error);
  - the cold distance at t = 0 is positive and larger than its own error estimate (otherwise the cold state already
    sits at the observable's equilibrium value: the comparison is degenerate, as the kappa = 0 energy comparison is);
  - D_hot(0) / D_cold(0) is below ROUNDOFF_RATIO = 1e12; a larger ratio means the cold distance is at round-off level
    relative to the hot one. The value 1e12 is a heuristic read off the stored main run (printed by
    prereg_v2/check_guard_on_main_run.py): the ratios there are at most 3.3e4 for KL, 1.9 for W1, 6.5e5 for Var(x) and
    1.4e4 for <U> at kappa != 0, while for <U> at kappa = 0 the 52 cases with a positive cold distance have ratios of
    at least 6.1e17 (the other 20 have a cold distance of zero), so any threshold in between separates them; it is NOT
    independent evidence.
"""

from __future__ import annotations

import numpy as np

from src.mpemba_analysis import Outcome, classify_metric

INVALID_INPUT = "INVALID_INPUT"
ROUNDOFF_RATIO = 1e12  # heuristic, see the module docstring


def check_curve_inputs(t: np.ndarray, d: np.ndarray, err: np.ndarray) -> list[str]:
    """Return a list of human-readable problems; an empty list means the input is a meaningful comparison."""
    t, d, err = np.asarray(t, float), np.asarray(d, float), np.asarray(err, float)
    problems: list[str] = []
    if t.ndim == 1 and t.size == 0:
        return ["empty time axis"]
    if t.ndim != 1 or d.shape != (t.size, 2) or err.shape != (t.size, 2):
        return [
            f"bad shapes: t {t.shape}, d {d.shape}, err {err.shape} (expected (n,), (n, 2), (n, 2))"
        ]
    if not (np.isfinite(t).all() and np.isfinite(d).all() and np.isfinite(err).all()):
        problems.append("non-finite values in t, d or err")
        return problems
    if t[0] != 0:
        problems.append("t[0] != 0 (P0 is evaluated at the start)")
    if not (np.diff(t) > 0).all():
        problems.append("t is not strictly increasing")
    if (err < 0).any():
        problems.append("negative error estimate")
    if (d < -err).any():
        problems.append(
            "distance below minus its own error estimate (not a rounding-level negative)"
        )
    if d[0, 0] <= 0:
        problems.append(
            "cold distance at t = 0 is not positive (degenerate: cold already at equilibrium)"
        )
    elif d[0, 0] <= err[0, 0]:
        problems.append(
            "cold distance at t = 0 does not exceed its own error estimate (degenerate comparison)"
        )
    elif d[0, 1] / d[0, 0] >= ROUNDOFF_RATIO:
        problems.append(
            f"hot/cold distance ratio at t = 0 is >= {ROUNDOFF_RATIO:.0e}: cold distance at round-off level (degenerate)"
        )
    return problems


def classify_metric_guarded(
    t: np.ndarray, d: np.ndarray, err: np.ndarray, t_max: float
) -> Outcome:
    """`classify_metric` after `check_curve_inputs`; invalid input gets the label INVALID_INPUT (all numbers nan)."""
    if check_curve_inputs(t, d, err):
        nan = float("nan")
        return Outcome(INVALID_INPUT, nan, nan, nan)
    return classify_metric(
        np.asarray(t, float), np.asarray(d, float), np.asarray(err, float), t_max
    )
