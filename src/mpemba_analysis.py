"""Curves, EFFECT classification and slow-mode overlap for PREREG_v2 (deviation D2).

Everything here implements definitions recorded in PREREG_v2.md D2 before this file existed.
Distances: KL and W1 to equilibrium (primary), |<U> - U_eq| and |Var(x) - Var_eq| (descriptive).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh_tridiagonal

from src.mpemba_exact import Grid, make_grid, out_rates, richardson, run_pair

D_FLOOR = 1e-18  # below this cold is equilibrated; times are excluded from the sustained test
MARGIN = 5.0  # crossing must exceed MARGIN * discretization error
P0_RATIO = 1.25  # hot must start at least this much farther than cold
SUSTAIN = 1.05  # the sustained window must start before t_max / SUSTAIN
PI_GOOD = 1e-20  # nodes with pi below this get a constant continuation of g1
METRICS = ("kl", "w1", "u", "var")


@dataclass
class Curves:
    """Distances at the finest dx, columns = (cold, hot) roles; errors are absolute."""

    t: np.ndarray
    d: dict[str, np.ndarray]  # Richardson-extrapolated distances, shape (n, 2)
    err: dict[str, np.ndarray]  # dx-difference + time-Richardson error, shape (n, 2)
    raw_coarse: dict[str, np.ndarray]  # raw coarse-run distances at the finest dx
    raw_fine: dict[str, np.ndarray]  # raw fine-run distances at the finest dx
    grid: Grid


def _series(grid: Grid, traj, name: str) -> np.ndarray:
    """Signed observable series (kl and w1 are already distances)."""
    if name == "kl":
        return traj.kl
    if name == "w1":
        return traj.w1
    if name == "u":
        return traj.u_mean - grid.u_eq
    return traj.var_x - grid.var_eq


def _as_distance(name: str, values: np.ndarray) -> np.ndarray:
    return np.abs(values) if name in ("u", "var") else values


def curves(
    energy_fn: Callable[[np.ndarray], np.ndarray],
    temperature: float,
    ic_fn: Callable[[Grid], tuple[np.ndarray, np.ndarray]],
    t_end: float,
    dxs: tuple[float, float] = (0.01, 0.005),
    half_width: float = 15.0,
    growth: float = 1.002,
    stride: int = 10,
) -> Curves:
    """Evolve the (cold, hot) pair at two spatial resolutions and assemble distances + errors."""
    per_dx = []
    for dx in dxs:
        grid = make_grid(energy_fn, temperature, half_width, dx)
        cold, hot = ic_fn(grid)
        coarse, fine = run_pair(grid, np.stack([cold, hot], axis=1), t_end, growth, stride)
        n = min(coarse.t.size, fine.t.size)
        if not np.allclose(coarse.t[:n], fine.t[:n], rtol=1e-9, atol=0):
            raise RuntimeError("coarse and fine time grids are misaligned")
        per_dx.append((grid, coarse, fine, n))
    n = min(p[3] for p in per_dx)
    t = per_dx[-1][1].t[:n]
    d, err, raw_c, raw_f = {}, {}, {}, {}
    for name in METRICS:
        extrapolated, time_err = [], []
        for grid, coarse, fine, _ in per_dx:
            c, f = _series(grid, coarse, name)[:n], _series(grid, fine, name)[:n]
            ex, e_t = richardson(c, f)
            extrapolated.append(_as_distance(name, ex))
            time_err.append(e_t)
        d[name] = extrapolated[-1]
        err[name] = np.abs(extrapolated[0] - extrapolated[-1]) + time_err[-1]
        grid, coarse, fine, _ = per_dx[-1]
        raw_c[name] = _as_distance(name, _series(grid, coarse, name)[:n])
        raw_f[name] = _as_distance(name, _series(grid, fine, name)[:n])
    return Curves(t=t, d=d, err=err, raw_coarse=raw_c, raw_fine=raw_f, grid=per_dx[-1][0])


@dataclass
class Outcome:
    label: str  # NO_TEST | NO_CROSSING | CROSSING_NO_MARGIN | EFFECT
    t_star: float  # start of the sustained-margin window (nan unless EFFECT)
    last_gap: float  # signed gap D_cold - D_hot at the last valid time (nan if none)
    p0_ratio: float  # D_hot(0) / D_cold(0)


def classify_metric(t: np.ndarray, d: np.ndarray, err: np.ndarray, t_max: float) -> Outcome:
    """D2.1 classification for one metric. Columns of d/err are (cold, hot); t[0] must be 0."""
    if t[0] != 0:
        raise ValueError("classify_metric requires t[0] == 0 (P0 is evaluated at the start)")
    d_c, d_h = d[:, 0], d[:, 1]
    p0_ratio = float(d_h[0] / d_c[0]) if d_c[0] > 0 else float("inf")
    if not d_h[0] >= P0_RATIO * d_c[0]:
        return Outcome("NO_TEST", float("nan"), float("nan"), p0_ratio)
    sel = (t > 0) & (t <= t_max) & (d_c >= D_FLOOR)
    if not sel.any():
        return Outcome("NO_CROSSING", float("nan"), float("nan"), p0_ratio)
    tt, gap = t[sel], (d_c - d_h)[sel]
    margin_err = MARGIN * (err[:, 0] + err[:, 1])[sel]
    last_gap = float(gap[-1])
    if not (gap > 0).any():
        return Outcome("NO_CROSSING", float("nan"), last_gap, p0_ratio)
    ok = gap > margin_err
    bad = np.flatnonzero(~ok)
    first_ok = 0 if bad.size == 0 else int(bad[-1]) + 1
    if ok[-1] and first_ok < tt.size and tt[first_ok] < t_max / SUSTAIN:
        return Outcome("EFFECT", float(tt[first_ok]), last_gap, p0_ratio)
    return Outcome("CROSSING_NO_MARGIN", float("nan"), last_gap, p0_ratio)


def slow_mode_overlaps(grid: Grid, rhos: list[np.ndarray]) -> tuple[float, list[float]]:
    """(lambda1, [c1(rho) ...]) with c1 = sum_i g1_i rho_i, g1 = phi1 / sqrt(pi) (left eigenvector).

    phi1 is the eigenvector of the symmetrised generator at -lambda1. Where pi < PI_GOOD the ratio
    is round-off dominated (phi1 ~ 1e-16 absolute), so g1 is continued by a constant from the
    nearest good node; g1 is bounded and flat there (drift-dominated tails). Requires the region
    pi >= PI_GOOD to be contiguous, i.e. moderate b/T (D2.4 restricts R to b/T <= 12).
    """
    n = grid.x.size
    diag = -out_rates(grid)
    off = np.sqrt(grid.up * grid.dn)
    w, v = eigh_tridiagonal(diag, off, select="i", select_range=(n - 2, n - 2))
    phi = v[:, 0]
    good = grid.pi > PI_GOOD
    idx = np.flatnonzero(good)
    if not np.all(np.diff(idx) == 1):
        raise ValueError("pi >= PI_GOOD region is not contiguous; R is undefined at this b/T")
    g = np.empty(n)
    g[good] = phi[good] / np.sqrt(grid.pi[good])
    g[: idx[0]] = g[idx[0]]
    g[idx[-1] + 1 :] = g[idx[-1]]
    return float(-w[0]), [float(np.sum(g * r)) for r in rhos]
