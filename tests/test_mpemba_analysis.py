"""Tests for src/mpemba_analysis.py (classification and slow-mode overlap, PREREG_v2 D2)."""

import numpy as np

from src.mpemba_analysis import classify_metric, slow_mode_overlaps
from src.mpemba_exact import cold_ic, gaussian_ic, lambda1_eig, make_grid, potential


def _curves(t, d_c, d_h, err_scale=0.0):
    d = np.stack([d_c, d_h], axis=1)
    err = err_scale * np.ones_like(d)
    return d, err


T_GRID = np.concatenate([[0.0], np.geomspace(1e-3, 100.0, 200)])


def test_effect_when_hot_starts_farther_and_stays_below():
    d_c = 0.7 * np.exp(-0.05 * T_GRID)
    d_h = 5.0 * np.exp(-1.0 * T_GRID) + 1e-3 * np.exp(-0.5 * T_GRID)
    d, err = _curves(T_GRID, d_c, d_h, err_scale=1e-9)
    out = classify_metric(T_GRID, d, err, t_max=100.0)
    assert out.label == "EFFECT"
    assert 0.0 < out.t_star < 100.0 / 1.05
    assert out.last_gap > 0


def test_no_test_when_hot_does_not_start_farther_by_the_ratio():
    d_c = 0.7 * np.exp(-0.05 * T_GRID)
    d_h = 0.8 * np.exp(-1.0 * T_GRID)  # 0.8 < 1.25 * 0.7
    d, err = _curves(T_GRID, d_c, d_h)
    assert classify_metric(T_GRID, d, err, t_max=100.0).label == "NO_TEST"


def test_no_crossing_when_hot_never_gets_closer():
    d_c = 0.7 * np.exp(-1.0 * T_GRID)
    d_h = 5.0 * np.exp(-0.5 * T_GRID)
    d, err = _curves(T_GRID, d_c, d_h)
    assert classify_metric(T_GRID, d, err, t_max=100.0).label == "NO_CROSSING"


def test_crossing_without_margin_is_not_an_effect():
    d_c = 0.7 * np.exp(-0.05 * T_GRID)
    d_h = 5.0 * np.exp(-1.0 * T_GRID) + 0.7 * np.exp(-0.05 * T_GRID) * 0.9999
    d, err = _curves(T_GRID, d_c, d_h, err_scale=1e-2)  # error larger than the tiny gap
    assert classify_metric(T_GRID, d, err, t_max=100.0).label == "CROSSING_NO_MARGIN"


def test_recrossing_at_the_end_is_not_sustained():
    d_c = 0.7 * np.exp(-0.05 * T_GRID)
    d_h = 5.0 * np.exp(-1.0 * T_GRID) + 1e-3 * np.exp(-0.5 * T_GRID)
    d_h = np.where(T_GRID > 60.0, 10.0 * d_c, d_h)  # hot jumps back above cold near the end
    d, err = _curves(T_GRID, d_c, d_h)
    assert classify_metric(T_GRID, d, err, t_max=100.0).label != "EFFECT"


TT = np.arange(0.0, 11.0)  # t = 0, 1, ..., 10 ; t_max = 10, so t_max / 1.05 = 9.52


def _pair(hot_late, cold=1.0, hot_first=2.0):
    """Cold constant at `cold`; hot starts at `hot_first` then follows `hot_late` (t = 1..10)."""
    d_c = np.full(TT.size, cold)
    d_h = np.concatenate([[hot_first], np.asarray(hot_late, dtype=float)])
    return np.stack([d_c, d_h], axis=1), np.zeros((TT.size, 2))


def test_t_star_is_the_first_record_after_the_last_bad_point_exactly():
    hot = np.full(10, 0.5)  # t = 1..10, hot below cold everywhere ...
    hot[3] = 1.2  # ... except a bad point at t = 4
    d, err = _pair(hot)
    out = classify_metric(TT, d, err, t_max=10.0)
    assert out.label == "EFFECT"
    assert out.t_star == 5.0  # an off-by-one (t* = 4) or a wrong side (t* = 3) must fail here


def test_window_that_starts_after_t_max_over_1_05_is_not_an_effect():
    hot = np.full(10, 0.5)
    hot[8] = 1.2  # bad at t = 9, so the sustained window would start at t = 10 >= 9.52
    d, err = _pair(hot)
    assert classify_metric(TT, d, err, t_max=10.0).label == "CROSSING_NO_MARGIN"


def test_records_after_t_max_are_ignored():
    t = np.arange(0.0, 13.0)  # records at t = 11, 12 lie beyond t_max = 10
    d_c = np.full(t.size, 1.0)
    d_h = np.concatenate([[2.0], np.full(12, 0.5)])
    d_h[[11, 12]] = 5.0  # bad, but beyond t_max: must not matter
    out = classify_metric(t, np.stack([d_c, d_h], axis=1), np.zeros((t.size, 2)), t_max=10.0)
    assert out.label == "EFFECT"
    d_h2 = np.concatenate([[2.0], np.full(12, 5.0)])
    d_h2[[1, 2, 11, 12]] = 0.5  # good early and beyond t_max, bad inside the window
    out2 = classify_metric(t, np.stack([d_c, d_h2], axis=1), np.zeros((t.size, 2)), t_max=10.0)
    assert out2.label == "CROSSING_NO_MARGIN"


def test_records_below_the_floor_are_excluded_from_the_sustained_test():
    d_c = np.concatenate([np.full(6, 1.0), np.full(5, 1e-20)])  # equilibrated after t = 5
    d_h = np.concatenate([[2.0], np.full(5, 0.5), np.full(5, 1e-19)])  # hot 'above' cold there
    out = classify_metric(TT, np.stack([d_c, d_h], axis=1), np.zeros((TT.size, 2)), t_max=10.0)
    assert out.label == "EFFECT"  # the floor-masked records (t = 6..10) must not count as bad


def test_degenerate_and_invalid_inputs_never_give_an_effect():
    d, err = _pair(np.full(10, 0.5))
    assert classify_metric(TT, d, err, t_max=0.0).label == "NO_CROSSING"  # empty selection
    d_nan, _ = _pair(np.concatenate([np.full(9, 0.5), [np.nan]]))
    assert classify_metric(TT, d_nan, err, t_max=10.0).label != "EFFECT"  # NaN at the last record
    zero = np.stack([np.zeros(TT.size), np.full(TT.size, 1.0)], axis=1)
    assert classify_metric(TT, zero, err, t_max=10.0).label == "NO_CROSSING"  # cold identically 0
    try:
        classify_metric(TT + 1.0, d, err, t_max=10.0)
    except ValueError:
        pass
    else:
        raise AssertionError("t[0] != 0 must raise")


def test_recrossing_at_the_end_is_exactly_crossing_without_margin():
    hot = np.concatenate([np.full(7, 0.5), np.full(3, 5.0)])  # hot closer, then far above again
    d, err = _pair(hot)
    assert classify_metric(TT, d, err, t_max=10.0).label == "CROSSING_NO_MARGIN"


def test_sustain_threshold_is_exactly_t_max_over_1_05():
    """Review P2: the 1.05 clause must be pinned (t_max / 1.05 = 9.5238 for t_max = 10)."""
    t = np.array([0.0, 9.0, 9.4, 9.6, 10.0])
    d_c = np.ones(5)
    err = np.zeros((5, 2))
    d_h = np.array([2.0, 5.0, 0.5, 0.5, 0.5])  # only t = 9.0 is bad -> window starts at 9.4
    out = classify_metric(t, np.stack([d_c, d_h], axis=1), err, t_max=10.0)
    assert out.label == "EFFECT" and out.t_star == 9.4
    d_h2 = np.array([2.0, 5.0, 5.0, 0.5, 0.5])  # 9.0 and 9.4 bad -> window starts at 9.6 > 9.5238
    out2 = classify_metric(t, np.stack([d_c, d_h2], axis=1), err, t_max=10.0)
    assert out2.label == "CROSSING_NO_MARGIN"  # would be EFFECT if the clause used t_max / 1.0


def test_p0_ratio_boundary_is_inclusive_at_1_25():
    d, err = _pair(np.full(10, 0.5), cold=1.0, hot_first=1.25)
    assert classify_metric(TT, d, err, t_max=10.0).label == "EFFECT"  # ratio exactly 1.25 passes
    d2, err2 = _pair(np.full(10, 0.5), cold=1.0, hot_first=1.2499)
    assert classify_metric(TT, d2, err2, t_max=10.0).label == "NO_TEST"


def test_gap_equal_to_the_margin_is_not_enough():
    d, _ = _pair(np.full(10, 0.5))  # gap = 0.5
    err = np.zeros((TT.size, 2))
    err[1:, 0] = 0.05
    err[1:, 1] = 0.05  # 5 * (0.05 + 0.05) = 0.5 exactly: the gap must EXCEED it
    assert classify_metric(TT, d, err, t_max=10.0).label == "CROSSING_NO_MARGIN"
    same = np.stack([np.ones(TT.size), np.concatenate([[2.0], np.ones(10)])], axis=1)
    assert classify_metric(TT, same, np.zeros((TT.size, 2)), t_max=10.0).label == "NO_CROSSING"


def test_error_margin_is_five_times_the_summed_errors():
    d, _ = _pair(np.full(10, 0.5))  # gap = 0.5 at every record
    err = np.zeros((TT.size, 2))
    err[1:, 0] = 0.049  # 5 * (0.049 + 0.049) = 0.49 < 0.5 -> EFFECT
    err[1:, 1] = 0.049
    assert classify_metric(TT, d, err, t_max=10.0).label == "EFFECT"
    err[1:, 1] = 0.06  # 5 * (0.049 + 0.06) = 0.545 > 0.5 -> no margin
    assert classify_metric(TT, d, err, t_max=10.0).label == "CROSSING_NO_MARGIN"


def test_slow_mode_overlap_continuation_holds_when_most_hot_mass_is_in_the_tails():
    """Coverage note from review (L3): hot sigma = 3 puts most mass where pi < 1e-20."""
    from src.mpemba_exact import evolve, time_grid

    grid = make_grid(lambda x: potential(x, 2.0, 0.05), 0.5, 12.0, 0.02)
    hot = gaussian_ic(grid, 3.0)
    assert hot[grid.pi < 1e-20].sum() > 0.3  # the continuation region carries real weight
    cold = cold_ic(grid)
    lam, (c_c, c_h) = slow_mode_overlaps(grid, [cold, hot])
    traj = evolve(
        grid, np.stack([cold, hot], axis=1), time_grid(4.0 / lam, 1.004), record_stride=20
    )
    predicted = 0.5 * np.array([c_c, c_h]) ** 2 * np.exp(-2.0 * lam * traj.t[-1])
    assert np.allclose(traj.kl[-1], predicted, rtol=0.05)


def test_slow_mode_overlap_predicts_late_time_kl_amplitude_with_tilt():
    """Independent check of c1: D_KL(t) ~ 0.5 * c1^2 * exp(-2 lambda1 t) once the fast modes died."""
    from src.mpemba_exact import evolve, time_grid

    grid = make_grid(lambda x: potential(x, 1.5, 0.05), 0.5, 8.0, 0.02)
    cold, hot = cold_ic(grid), gaussian_ic(grid, 1.5, 0.3)
    lam, (c_c, c_h) = slow_mode_overlaps(grid, [cold, hot])
    t_late = 4.0 / lam
    traj = evolve(grid, np.stack([cold, hot], axis=1), time_grid(t_late, 1.004), record_stride=20)
    t, kl = traj.t[-1], traj.kl[-1]
    predicted = 0.5 * np.array([c_c, c_h]) ** 2 * np.exp(-2.0 * lam * t)
    assert np.allclose(kl, predicted, rtol=0.05)
    assert abs(c_h) < abs(c_c)  # the hot state overlaps less with the slow mode


def test_slow_mode_overlap_is_linear_and_odd_for_symmetric_potential():
    grid = make_grid(lambda x: potential(x, 3.0, 0.0), 0.3, 8.0, 0.01)
    right = cold_ic(grid)
    left = right[::-1].copy()  # mirror: grid is symmetric about 0
    sym = gaussian_ic(grid, 2.0)
    lam, (c_r, c_l, c_s, c_mix) = slow_mode_overlaps(
        grid, [right, left, sym, 0.7 * right + 0.3 * left]
    )
    assert (
        abs(lam / lambda1_eig(make_grid(lambda x: potential(x, 3.0, 0.0), 0.3, 4.0, 0.01)) - 1)
        < 0.05
    )
    assert abs(abs(c_r) - 1.0) < 0.05  # two-state limit: g1 ~ sign(x)
    assert abs(c_r + c_l) < 1e-6  # odd mode: mirror flips the sign
    assert abs(c_s) < 1e-6  # symmetric state has no slow-mode component
    assert abs(c_mix - (0.7 * c_r + 0.3 * c_l)) < 1e-9  # linearity
