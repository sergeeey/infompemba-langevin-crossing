"""Exact 1D Fokker-Planck evolution for pre-registration v2 (PREREG_v2.md, deviation D1).

Scharfetter-Gummel finite-volume generator + implicit Euler on a geometric time grid.

WHY not the spectral solver in fokker_planck.py: projecting an initial state onto the modes needs
exp(U/2T); for the hot Gaussian tails this leaves float64 range (U/2T ~ 1e4 at |x|=4.2 for
b=5, T=0.05). This scheme never forms exp(U/2T), conserves mass and its discrete stationary state
is exactly the Boltzmann weight exp(-U_i/T).

WHY a hand-written subtraction-free solve instead of scipy.linalg.solve_banded: at large time
steps I - hQ has condition number ~ h * lambda_max (1e10 .. 1e50). Ordinary Gaussian elimination
then loses the slow modes (first run of gate V2: mass drift 3e-8, symmetric hot state acquired a
spurious slow-mode component). For an M-matrix the Thomas recursion can be rewritten so every
update adds positive numbers only, which keeps componentwise relative accuracy at any h and makes
rho >= 0 hold by construction.

Dynamics: d rho/dt = d/dx (T d rho/dx + U' rho), reflecting walls at +-half_width.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import xlogy

# WHY: below this equilibrium weight the ratio rho/pi is not representable; use the log form.
_PI_FLOOR = 1e-250


def potential(x: np.ndarray, barrier: float, kappa: float = 0.0) -> np.ndarray:
    """U(x) = b (x^2 - 1)^2 + kappa * b * x  (kappa tilts the wells)."""
    return barrier * (x**2 - 1.0) ** 2 + kappa * barrier * x


def bernoulli(z: np.ndarray) -> np.ndarray:
    """B(z) = z / (exp(z) - 1), stable for large |z| (B(+big) -> 0, B(-big) -> |z|)."""
    z = np.asarray(z, dtype=float)
    small = np.abs(z) < 1e-8
    safe = np.where(small, 1.0, z)
    with np.errstate(over="ignore"):
        val = safe / np.expm1(safe)
    return np.where(small, 1.0 - z / 2.0, val)


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    dx: float
    temperature: float
    energy: np.ndarray
    up: np.ndarray  # rate i -> i+1, length N-1
    dn: np.ndarray  # rate i+1 -> i, length N-1
    logpi: np.ndarray
    pi: np.ndarray
    cdf_pi: np.ndarray
    u_eq: float
    var_eq: float


def make_grid(
    energy_fn: Callable[[np.ndarray], np.ndarray], temperature: float, half_width: float, dx: float
) -> Grid:
    """Uniform grid symmetric about x=0 (a node sits at 0)."""
    n = round(half_width / dx)
    x = dx * np.arange(-n, n + 1)
    u = energy_fn(x)
    delta = (u[1:] - u[:-1]) / temperature
    scale = temperature / dx**2
    up = scale * bernoulli(delta)
    dn = scale * bernoulli(-delta)
    logpi = -u / temperature
    logpi = logpi - (logpi.max() + np.log(np.sum(np.exp(logpi - logpi.max()))))
    pi = np.exp(logpi)
    mean = float(np.sum(pi * x))
    return Grid(
        x=x,
        dx=dx,
        temperature=temperature,
        energy=u,
        up=up,
        dn=dn,
        logpi=logpi,
        pi=pi,
        cdf_pi=np.cumsum(pi),
        u_eq=float(np.sum(pi * u)),
        var_eq=float(np.sum(pi * x**2) - mean**2),
    )


def out_rates(grid: Grid) -> np.ndarray:
    out = np.zeros(grid.x.size)
    out[:-1] += grid.up
    out[1:] += grid.dn
    return out


def apply_q(grid: Grid, rho: np.ndarray) -> np.ndarray:
    """(Q rho)_i = dn_i rho_{i+1} + up_{i-1} rho_{i-1} - out_i rho_i."""
    res = -out_rates(grid) * rho
    res[1:] += grid.up * rho[:-1]
    res[:-1] += grid.dn * rho[1:]
    return res


@njit(cache=True)
def _implicit_step(up: np.ndarray, dn: np.ndarray, h: float, rho: np.ndarray) -> None:
    """Solve (I - hQ) x = rho in place, subtraction-free (columns of rho are independent states).

    With d_i the eliminated diagonal and e_i = d_i - h*up_i:
        d_0 = 1 + h*up_0,  e_0 = 1
        d_i = 1 + h*up_i + h*dn_{i-1}*e_{i-1}/d_{i-1},   e_i = 1 + h*dn_{i-1}*e_{i-1}/d_{i-1}
    Forward/backward substitution likewise only adds positive terms.
    """
    n, m = rho.shape
    d = np.empty(n)
    d[0] = 1.0 + h * up[0]
    e_prev = 1.0
    for i in range(1, n):
        t = h * dn[i - 1] * (e_prev / d[i - 1])
        up_i = up[i] if i < n - 1 else 0.0
        d[i] = 1.0 + h * up_i + t
        e_prev = 1.0 + t
    for j in range(m):
        for i in range(1, n):
            rho[i, j] += h * up[i - 1] / d[i - 1] * rho[i - 1, j]
        rho[n - 1, j] /= d[n - 1]
        for i in range(n - 2, -1, -1):
            rho[i, j] = (rho[i, j] + h * dn[i] * rho[i + 1, j]) / d[i]


def _phi(r: np.ndarray) -> np.ndarray:
    """phi(r) = (1+r) ln(1+r) - r >= 0 without cancellation.

    WHY a series for small |r|: the direct form subtracts two numbers ~ r, leaving an absolute
    error ~1e-16 per node; summed with weights pi that is a KL noise floor ~1e-17 (gate V2(d)
    first failed with 2.25e-17). phi(r) = sum_{k>=2} (-1)^k r^k / (k (k-1)).
    """
    out = np.empty_like(r)
    small = np.abs(r) < 0.05
    rs = r[small]
    acc = np.zeros_like(rs)
    power = rs * rs
    sign = 1.0
    for k in range(2, 14):
        acc += sign * power / (k * (k - 1))
        power = power * rs
        sign = -sign
    out[small] = acc
    big = r[~small]
    out[~small] = xlogy(1.0 + big, 1.0 + big) - big
    return out


def kl_div(grid: Grid, rho: np.ndarray) -> float:
    """sum_i [rho ln(rho/pi) - rho + pi] = sum_i pi_i phi(rho_i/pi_i - 1); every term >= 0."""
    mask = grid.pi > _PI_FLOOR
    pi_m = grid.pi[mask]
    r = (rho[mask] - pi_m) / pi_m
    total = float(np.sum(pi_m * _phi(r)))
    tail = ~mask & (rho > 0)
    if tail.any():
        rt = rho[tail]
        total += float(np.sum(rt * (np.log(rt) - grid.logpi[tail]) - rt))
    return total


def w1_dist(grid: Grid, rho: np.ndarray) -> float:
    """1D Wasserstein-1 distance to equilibrium via CDFs (measures live on grid nodes)."""
    return float(np.sum(np.abs(np.cumsum(rho) - grid.cdf_pi)) * grid.dx)


def observables(grid: Grid, rho: np.ndarray) -> tuple[float, float]:
    """<U> and Var(x)."""
    mean = float(np.sum(rho * grid.x))
    return float(np.sum(rho * grid.energy)), float(np.sum(rho * grid.x**2) - mean**2)


def gaussian_ic(grid: Grid, sigma: float, centre: float = 0.0) -> np.ndarray:
    w = np.exp(-0.5 * ((grid.x - centre) / sigma) ** 2)
    return w / w.sum()


def cold_ic(grid: Grid) -> np.ndarray:
    """Right-well local equilibrium: Boltzmann restricted to x > 0 (half weight at x = 0)."""
    w = np.where(grid.x > 0, grid.pi, 0.0)
    w[grid.x.size // 2] = 0.5 * grid.pi[grid.x.size // 2]
    return w / w.sum()


@dataclass
class Trajectory:
    """Recorded metrics; column j corresponds to initial condition j."""

    t: np.ndarray
    kl: np.ndarray
    w1: np.ndarray
    u_mean: np.ndarray
    var_x: np.ndarray
    mass_drift: float
    min_rho: float
    final_rho: np.ndarray


def time_grid(t_end: float, growth: float, t_min: float = 1e-9) -> np.ndarray:
    """0, t_min, t_min*g, ... up to (and past) t_end."""
    steps = int(np.ceil(np.log(t_end / t_min) / np.log(growth))) + 1
    return np.concatenate([[0.0], t_min * growth ** np.arange(steps + 1)])


def evolve(
    grid: Grid,
    rho0: np.ndarray,
    times: np.ndarray,
    record_stride: int = 1,
    check_every_step: bool = False,
) -> Trajectory:
    """Implicit-Euler evolution of the columns of rho0 (shape (N, m)) over `times`.

    Records at step 0 and at steps whose 1-based index k satisfies (k - 1) % record_stride == 0.
    Mass drift and min rho are tracked at recorded steps only, or at every step when
    `check_every_step` is set (validation gates use that; production runs skip the extra cost).
    """
    rho = np.array(rho0, dtype=float, copy=True)
    if rho.ndim == 1:
        rho = rho[:, None]
    m = rho.shape[1]
    rec_t, rec_kl, rec_w1, rec_u, rec_v = [], [], [], [], []
    drift = 0.0
    min_rho = float(rho.min())

    def record(t: float) -> None:
        rec_t.append(t)
        rec_kl.append([kl_div(grid, rho[:, j]) for j in range(m)])
        rec_w1.append([w1_dist(grid, rho[:, j]) for j in range(m)])
        uv = [observables(grid, rho[:, j]) for j in range(m)]
        rec_u.append([a for a, _ in uv])
        rec_v.append([b for _, b in uv])

    record(0.0)
    for k in range(1, times.size):
        _implicit_step(grid.up, grid.dn, float(times[k] - times[k - 1]), rho)
        recorded = (k - 1) % record_stride == 0
        if check_every_step or recorded:
            min_rho = min(min_rho, float(rho.min()))
            drift = max(drift, float(np.abs(rho.sum(axis=0) - 1.0).max()))
        if recorded:
            record(float(times[k]))
    return Trajectory(
        t=np.array(rec_t),
        kl=np.array(rec_kl),
        w1=np.array(rec_w1),
        u_mean=np.array(rec_u),
        var_x=np.array(rec_v),
        mass_drift=drift,
        min_rho=min_rho,
        final_rho=rho,
    )


def richardson(coarse: np.ndarray, fine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First-order Richardson: (extrapolated, error estimate |fine - extrapolated|).

    `coarse` uses time ratio q, `fine` uses sqrt(q); they must be sampled at the same times.
    """
    extrap = 2.0 * fine - coarse
    return extrap, np.abs(fine - extrap)


def run_pair(
    grid: Grid, rho0: np.ndarray, t_end: float, growth: float = 1.002, stride: int = 10
) -> tuple[Trajectory, Trajectory]:
    """Coarse (ratio q) and fine (ratio sqrt q) runs sampled at common times.

    Coarse step k has t = t_min q^(k-1); fine step j has t = t_min q^((j-1)/2); they coincide when
    j = 2k - 1, so recording coarse every `stride` steps and fine every `2*stride` keeps alignment.
    """
    coarse = evolve(grid, rho0, time_grid(t_end, growth), record_stride=stride)
    fine = evolve(grid, rho0, time_grid(t_end, np.sqrt(growth)), record_stride=2 * stride)
    return coarse, fine


def lambda1_estimate(
    energy_fn: Callable[[np.ndarray], np.ndarray],
    temperature: float,
    half_width: float = 6.0,
    dx: float = 1e-3,
) -> float:
    """Slowest relaxation rate ~ k_R + k_L with k = 1 / (2 tau_top), tau_top = mean first-passage
    time from a well minimum to the barrier top (reflecting far wall), by exact quadrature in the
    log domain (no exp overflow). Asymptotic for b/T >> 1; cross-checked against eigvalsh in V3.
    """
    n = round(half_width / dx)
    x = dx * np.arange(-n, n + 1)
    u = energy_fn(x)
    inner = (x > -1.0) & (x < 1.0)
    top = int(np.flatnonzero(inner)[np.argmax(u[inner])])
    right = top + int(np.argmin(u[top:]))
    left = int(np.argmin(u[: top + 1]))
    lw = -u / temperature + np.log(dx)
    log_int_right = np.logaddexp.accumulate(lw[::-1])[::-1]  # log int_y^L e^{-U/T}
    log_int_left = np.logaddexp.accumulate(lw)  # log int_{-L}^y e^{-U/T}

    def log_tau(a: int, b_idx: int, log_inner: np.ndarray) -> float:
        seg = slice(a, b_idx + 1)
        terms = u[seg] / temperature + log_inner[seg] + np.log(dx)
        mx = terms.max()
        return float(mx + np.log(np.sum(np.exp(terms - mx))) - np.log(temperature))

    tau_r = np.exp(log_tau(top, right, log_int_right))
    tau_l = np.exp(log_tau(left, top, log_int_left))
    return 1.0 / (2.0 * tau_r) + 1.0 / (2.0 * tau_l)


def lambda1_eig(grid: Grid) -> float:
    """Slowest nonzero rate from the symmetrised generator (reliable only when lambda1 is not
    swamped by float64 round-off relative to the generator norm)."""
    from scipy.linalg import eigh_tridiagonal

    diag = -out_rates(grid)
    off = np.sqrt(grid.up * grid.dn)
    n = diag.size
    vals = eigh_tridiagonal(diag, off, select="i", select_range=(n - 2, n - 1), eigvals_only=True)
    return float(-vals[0])
