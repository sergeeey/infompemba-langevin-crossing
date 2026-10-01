"""Guards around classify_metric: invalid or degenerate input is not labelled, valid input is passed through unchanged."""

from pathlib import Path

import numpy as np
import pytest

from src.mpemba_analysis import classify_metric
from src.mpemba_guards import INVALID_INPUT, check_curve_inputs, classify_metric_guarded

T = np.array([0.0, 1.0, 2.0, 3.0])
ERR = np.full((4, 2), 1e-9)


def _curves(d_cold, d_hot):
    return np.column_stack([d_cold, d_hot])


def test_documented_limitation_unguarded_degenerate_input_is_labelled_effect():
    # both states at equilibrium at t = 0 (distance 0); later the "hot" distance is smaller: non-strict P0 lets it through
    d = _curves([0.0, 1e-3, 1e-3, 1e-3], [0.0, 1e-4, 1e-4, 1e-4])
    assert (
        classify_metric(T, d, ERR, 10.0).label == "EFFECT"
    )  # the limitation the guard exists for
    out = classify_metric_guarded(T, d, ERR, 10.0)
    assert out.label == INVALID_INPUT and np.isnan(out.p0_ratio)
    assert any("not positive" in p for p in check_curve_inputs(T, d, ERR))


def test_valid_crossing_passes_through_unchanged():
    d = _curves([0.1, 0.05, 0.02, 0.01], [0.5, 0.01, 0.001, 0.0001])
    a, b = classify_metric(T, d, ERR, 10.0), classify_metric_guarded(T, d, ERR, 10.0)
    assert check_curve_inputs(T, d, ERR) == []
    assert (a.label, a.p0_ratio) == (b.label, b.p0_ratio)


def test_non_finite_and_bad_shape_and_time_axis_are_reported():
    d = _curves([0.1, 0.05, 0.02, 0.01], [0.5, 0.01, 0.001, 0.0001])
    bad = d.copy()
    bad[2, 1] = np.nan
    assert any("non-finite" in p for p in check_curve_inputs(T, bad, ERR))
    assert any("bad shapes" in p for p in check_curve_inputs(T, d[:, :1], ERR))
    assert any("t[0]" in p for p in check_curve_inputs(T + 1.0, d, ERR))
    assert any(
        "strictly increasing" in p
        for p in check_curve_inputs(np.array([0.0, 1.0, 1.0, 2.0]), d, ERR)
    )


def test_empty_input_is_reported_not_raised():
    assert check_curve_inputs(np.array([]), np.empty((0, 2)), np.empty((0, 2))) == [
        "empty time axis"
    ]


def test_small_negative_within_error_is_allowed_but_large_negative_is_not():
    d = _curves([0.1, 0.05, 0.02, 0.01], [0.5, 0.01, 0.001, 0.0001])
    ok = d.copy()
    ok[
        3, 1
    ] = (
        -0.5e-9
    )  # within its own error estimate (1e-9): a Richardson rounding-level negative
    assert check_curve_inputs(T, ok, ERR) == []
    bad = d.copy()
    bad[3, 1] = -1e-6
    assert any("below minus" in p for p in check_curve_inputs(T, bad, ERR))


def test_cold_distance_not_above_its_error_is_degenerate():
    d = _curves([1e-12, 1e-12, 1e-12, 1e-12], [0.5, 0.1, 0.01, 0.001])
    err = np.full((4, 2), 1e-9)
    assert any(
        "does not exceed its own error" in p for p in check_curve_inputs(T, d, err)
    )


def test_cold_distance_at_roundoff_level_relative_to_hot_is_degenerate():
    d = _curves([3e-17, 3e-17, 3e-17, 3e-17], [0.5, 0.1, 0.01, 0.001])
    err = np.full(
        (4, 2), 1e-20
    )  # above its error estimate, but 1.7e16 times smaller than the hot distance
    assert any("round-off level" in p for p in check_curve_inputs(T, d, err))


@pytest.mark.skipif(
    not Path("prereg_v2/out").exists(),
    reason="curve files are not in the git tree (see REPRODUCE.md)",
)
@pytest.mark.parametrize(
    "name", ["b2.0_T0.2_k0.0", "b1.0_T0.1_k0.1", "b0.1_T1.0_k0.1", "b5.0_T0.05_k0.02"]
)
def test_guard_agrees_with_classifier_on_stored_main_run_curves(name):
    import pandas as pd

    z = np.load(f"prereg_v2/out/{name}.npz")
    row = pd.read_csv("prereg_v2/summary_main.csv")
    b, t, k = (float(x[1:]) for x in name.split("_"))
    r = row[(row.b == b) & (row["T"] == t) & (row.kappa == k)].iloc[0]
    for metric in ("kl", "w1"):
        a = classify_metric(z["t"], z[f"d_{metric}"], z[f"err_{metric}"], r.t_max)
        g = classify_metric_guarded(
            z["t"], z[f"d_{metric}"], z[f"err_{metric}"], r.t_max
        )
        assert (a.label, a.p0_ratio) == (
            g.label,
            g.p0_ratio,
        )  # the guard is a pure pass-through
        # the CSV stores the numbers rounded to ~16 digits, so compare with a tolerance
        assert a.label == r[f"{metric}_label"]
        assert a.p0_ratio == pytest.approx(r[f"{metric}_p0"], rel=1e-12)
