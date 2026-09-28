"""Unit tests for src/mpemba_exact.py (solver for PREREG_v2.md, deviation D1)."""

import numpy as np

from src.mpemba_exact import (
    _implicit_step,
    _phi,
    apply_q,
    bernoulli,
    cold_ic,
    evolve,
    gaussian_ic,
    kl_div,
    lambda1_eig,
    lambda1_estimate,
    make_grid,
    potential,
    run_pair,
    time_grid,
    w1_dist,
)


def _grid(barrier=1.0, temperature=0.3, kappa=0.0, half_width=5.0, dx=0.02):
    return make_grid(lambda x: potential(x, barrier, kappa), temperature, half_width, dx)


def test_bernoulli_limits_and_no_nan():
    z = np.array([-800.0, -1.0, 0.0, 1e-12, 1.0, 800.0])
    b = bernoulli(z)
    assert np.all(np.isfinite(b))
    assert b[2] == 1.0
    assert abs(b[0] - 800.0) < 1e-9
    assert b[-1] == 0.0


def test_discrete_equilibrium_is_stationary():
    grid = _grid(kappa=0.05)
    res = np.abs(apply_q(grid, grid.pi))
    assert res.max() < 1e-12


def test_implicit_step_preserves_equilibrium_and_mass_at_huge_step():
    grid = _grid()
    rho = np.stack([grid.pi, gaussian_ic(grid, 1.0)], axis=1)
    for h in (1e-9, 1.0, 1e12, 1e40):
        _implicit_step(grid.up, grid.dn, h, rho)
    assert np.abs(rho.sum(axis=0) - 1.0).max() < 1e-12
    assert rho.min() >= 0.0
    assert np.abs(rho[:, 0] - grid.pi).max() < 1e-12  # equilibrium column untouched


def test_phi_series_matches_direct_form_at_the_switch():
    from scipy.special import xlogy

    r = np.array([0.0499, 0.0501, -0.0499, -0.0501, 0.5])
    direct = xlogy(1 + r, 1 + r) - r
    assert np.allclose(_phi(r), direct, rtol=1e-9, atol=1e-15)
    assert np.all(_phi(np.array([1e-9, -1e-9])) > 0.0)


def test_kl_and_w1_vanish_at_equilibrium_and_cold_kl_is_ln2():
    grid = _grid()
    assert kl_div(grid, grid.pi) < 1e-25
    assert w1_dist(grid, grid.pi) < 1e-14
    assert abs(kl_div(grid, cold_ic(grid)) - np.log(2.0)) < 1e-3


def test_relaxation_toward_equilibrium_is_monotone_for_cold():
    grid = _grid(barrier=0.5, temperature=0.5)
    traj = evolve(grid, cold_ic(grid), time_grid(50.0, 1.05), record_stride=5)
    kl = traj.kl[:, 0]
    assert np.all(np.diff(kl) <= 1e-12)
    assert kl[-1] < 1e-3 * kl[0]


def test_richardson_pair_is_more_accurate_than_coarse_on_harmonic_ou():
    temperature, m0, s0, t_end = 0.5, 1.0, 0.4, 3.0
    grid = make_grid(lambda x: 0.5 * x**2, temperature, 6.0, 0.02)
    coarse, fine = run_pair(grid, gaussian_ic(grid, s0, m0), t_end, growth=1.02, stride=5)
    n = min(coarse.t.size, fine.t.size)
    t = coarse.t[:n]
    assert np.allclose(t, fine.t[:n], rtol=1e-9, atol=0), "coarse/fine time grids misaligned"
    var = temperature + (s0**2 - temperature) * np.exp(-2 * t)
    mean = m0 * np.exp(-t)
    exact = 0.5 * (var / temperature + mean**2 / temperature - 1 - np.log(var / temperature))
    sel = exact > 1e-4
    err_c = np.abs(coarse.kl[:n, 0][sel] - exact[sel]) / exact[sel]
    err_f = np.abs(fine.kl[:n, 0][sel] - exact[sel]) / exact[sel]
    extrap = 2 * fine.kl[:n, 0][sel] - coarse.kl[:n, 0][sel]
    err_x = np.abs(extrap - exact[sel]) / exact[sel]
    assert err_f.max() < err_c.max()
    assert err_x.max() < err_f.max()


def test_symmetric_hot_has_no_spurious_slow_mode_component():
    """kappa=0, symmetric hot state: late-time D must fall far below the cold state's D."""
    grid = _grid(barrier=2.0, temperature=0.3, half_width=6.0, dx=0.02)
    rho0 = np.stack([cold_ic(grid), gaussian_ic(grid, 3.0)], axis=1)
    lam1 = lambda1_estimate(lambda x: potential(x, 2.0, 0.0), 0.3)
    traj = evolve(grid, rho0, time_grid(3.0 / lam1, 1.02), record_stride=5)
    assert traj.kl[-1, 1] < 1e-6 * traj.kl[-1, 0]


def test_lambda1_quadrature_agrees_with_eigvalsh_in_resolvable_regime():
    grid = _grid(barrier=1.0, temperature=0.3, half_width=4.0, dx=0.01)
    est = lambda1_estimate(lambda x: potential(x, 1.0, 0.0), 0.3)
    assert 1 / 1.5 < est / lambda1_eig(grid) < 1.5


def test_implicit_step_matches_dense_solve_at_moderate_h():
    grid = _grid(barrier=1.0, temperature=0.4, kappa=0.05, half_width=3.0, dx=0.05)
    n = grid.x.size
    q = np.zeros((n, n))
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        q[:, i] = apply_q(grid, e)
    rho0 = np.stack([gaussian_ic(grid, 0.8, 0.5), cold_ic(grid)], axis=1)
    for h in (1e-3, 0.1, 10.0):
        expected = np.linalg.solve(np.eye(n) - h * q, rho0)
        got = rho0.copy()
        _implicit_step(grid.up, grid.dn, h, got)
        assert np.abs(got - expected).max() < 1e-10 * max(1.0, expected.max())


def test_kl_tail_branch_agrees_with_direct_formula(monkeypatch):
    """Nodes below the pi floor use the log form; result must match the direct sum."""
    grid = _grid(barrier=2.0, temperature=0.1, half_width=4.0, dx=0.02)
    rho = gaussian_ic(grid, 1.5)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        direct = np.where(
            (rho > 0) & (grid.pi > 0), rho * np.log(rho / grid.pi) - rho + grid.pi, 0.0
        )
    mask = grid.pi > 1e-30
    assert (~mask).any(), "test needs nodes below the floor"
    reference = float(direct[mask].sum())
    monkeypatch.setattr("src.mpemba_exact._PI_FLOOR", 1e-30)
    tail_terms = rho[~mask] * (np.log(rho[~mask]) - grid.logpi[~mask]) - rho[~mask]
    assert abs(kl_div(grid, rho) - (reference + float(tail_terms.sum()))) < 1e-9 * abs(reference)


def test_kl_is_invariant_to_the_pi_floor_up_to_dropped_tail_pi(monkeypatch):
    """Independent of the tail formula: the floor only picks the branch, not the value."""
    grid = _grid(barrier=2.0, temperature=0.1, half_width=4.0, dx=0.02)
    rho = gaussian_ic(grid, 1.5)
    default = kl_div(grid, rho)
    monkeypatch.setattr("src.mpemba_exact._PI_FLOOR", 1e-30)
    patched = kl_div(grid, rho)
    assert (grid.pi <= 1e-30).any()
    assert abs(patched - default) < 1e-9 * abs(default)


def test_langevin_is_reproducible_for_a_fixed_seed():
    """Per-particle seeding (D1.4): same seed -> identical paths, different seed -> different."""
    from run_gates_v2_langevin import _langevin

    x0 = np.linspace(-1.5, 1.5, 4000)
    rec = np.array([10, 60, 200], dtype=np.int64)
    first = _langevin(x0, 1.0, 0.3, 0.002, rec, 123)
    again = _langevin(x0, 1.0, 0.3, 0.002, rec, 123)
    other = _langevin(x0, 1.0, 0.3, 0.002, rec, 9999)
    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_w1_of_point_mass_equals_mean_distance():
    grid = _grid(barrier=1.0, temperature=0.5)
    j = grid.x.size // 2 + 37
    rho = np.zeros(grid.x.size)
    rho[j] = 1.0
    expected = float(np.sum(grid.pi * np.abs(grid.x - grid.x[j])))
    assert abs(w1_dist(grid, rho) - expected) < 1e-12


def test_tilted_potential_relaxes_to_tilted_equilibrium():
    grid = _grid(barrier=1.0, temperature=0.5, kappa=0.1, half_width=4.0, dx=0.02)
    lam = lambda1_eig(grid)
    traj = evolve(grid, cold_ic(grid), time_grid(30.0 / lam, 1.05), record_stride=5)
    assert np.abs(traj.final_rho[:, 0] - grid.pi).sum() < 1e-8
    assert traj.mass_drift < 1e-12
    # tilt breaks the symmetry: equilibrium mass is not split evenly between the wells
    left = float(grid.pi[grid.x < 0].sum())
    assert abs(left - 0.5) > 0.01
