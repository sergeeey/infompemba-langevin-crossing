"""
Fokker-Planck spectral analysis for 1D double-well potential.

Potential: U(x) = barrier * (x^2 - 1)^2
FP operator: L f = T f'' + (U' f)'  (probability form)
           = T f'' + U'' f + U' f'

Eigenvalues: 0 = λ₀ > -λ₁ > -λ₂ > ...
  λ₁ = inter-basin (slow, Kramers)
  λ₂ = intra-basin (fast)

Mpemba condition: λ₁ ≪ λ₂  (timescale separation)

Also provides Kramers analytical approximation for comparison.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def build_fp_operator(
    barrier: float, T: float, x_min: float = -4.0, x_max: float = 4.0, n_pts: int = 401
) -> tuple:
    """
    Build discretized FP operator in similarity-transformed form.

    The FP equation ∂ₜp = Lp with L = T∂²/∂x² + ∂/∂x(U'p) is
    non-symmetric. We use the similarity transform:

        H = e^{U/2T} L e^{-U/2T}

    which gives the symmetric operator (Schrodinger form):

        H f = T f'' - V_eff f
        V_eff(x) = (U')²/(4T) - U''/2

    Eigenvalues of H = eigenvalues of L. H is real symmetric → clean numerics.

    Returns: (H_sparse, x_grid, dx)
    """
    x = np.linspace(x_min, x_max, n_pts)
    dx = x[1] - x[0]

    # Potential derivatives
    # U = barrier * (x^2 - 1)^2
    U_prime = 4.0 * barrier * x * (x**2 - 1.0)
    U_double_prime = 4.0 * barrier * (3.0 * x**2 - 1.0)

    # Effective quantum potential
    V_eff = U_prime**2 / (4.0 * T) - U_double_prime / 2.0

    # Kinetic: T * d²/dx² via central differences
    diag_main = -2.0 * T / dx**2 - V_eff
    diag_off = T / dx**2

    # Build tridiagonal sparse matrix
    H = sparse.diags(
        [diag_off, diag_main, diag_off],
        offsets=[-1, 0, 1],
        shape=(n_pts, n_pts),
        format="csr",
    )

    # Absorbing boundary: already handled by finite grid (wavefunction → 0 at edges)
    return H, x, dx


def compute_eigenvalues(
    barrier: float, T: float, n_eigs: int = 6, n_pts: int = 401, x_range: tuple = None
) -> dict:
    """
    Compute lowest eigenvalues of the FP operator.

    Returns dict with eigenvalues, spectral gap, timescale ratio, etc.
    """
    # WHY: auto-expand grid for high barriers to avoid boundary artifacts
    if x_range is None:
        half_width = max(4.0, 2.0 + np.sqrt(barrier))
        x_range = (-half_width, half_width)

    H, x, dx = build_fp_operator(barrier, T, x_range[0], x_range[1], n_pts)

    # WHY: shift-invert (sigma=0) finds eigenvalues nearest 0 — exactly the top
    # of the spectrum for this negative-semidefinite operator. 'LA' fails to converge
    # on stiff problems because ARPACK needs to resolve tiny spectral gaps.
    n_eigs_actual = min(n_eigs, H.shape[0] - 2)
    eigenvalues, eigenvectors = eigsh(H, k=n_eigs_actual, which="LM", sigma=0.0, maxiter=10000)

    # Sort descending (λ₀ ≈ 0 first, then increasingly negative)
    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # λ₀ should be ≈ 0 (ground state). Shift so λ₀ = 0 exactly.
    eigenvalues = eigenvalues - eigenvalues[0]

    # Rates: λ_k (positive) = -eigenvalue_k
    lambda_0 = 0.0
    lambda_1 = -eigenvalues[1] if len(eigenvalues) > 1 else np.nan
    lambda_2 = -eigenvalues[2] if len(eigenvalues) > 2 else np.nan

    # Safety: rates must be positive
    lambda_1 = max(lambda_1, 1e-30) if not np.isnan(lambda_1) else np.nan
    lambda_2 = max(lambda_2, 1e-30) if not np.isnan(lambda_2) else np.nan

    gap_ratio = lambda_1 / lambda_2 if lambda_2 > 1e-15 else np.nan
    tau_1 = 1.0 / lambda_1 if lambda_1 > 1e-15 else np.inf
    tau_2 = 1.0 / lambda_2 if lambda_2 > 1e-15 else np.inf

    return {
        "barrier": barrier,
        "T": T,
        "lambda_0": lambda_0,
        "lambda_1": lambda_1,
        "lambda_2": lambda_2,
        "gap_ratio": gap_ratio,  # λ₁/λ₂ < 1 means timescale separation
        "tau_1": tau_1,  # slow (inter-basin) timescale
        "tau_2": tau_2,  # fast (intra-basin) timescale
        "timescale_ratio": tau_1 / tau_2 if tau_2 > 1e-15 else np.inf,
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "x_grid": x,
    }


def kramers_rate(barrier: float, T: float) -> float:
    """
    Kramers escape rate for U(x) = barrier*(x^2-1)^2.

    Wells at x = ±1: U''(±1) = 8*barrier  →  ω_well = sqrt(8*barrier)
    Saddle at x = 0: U(0) = barrier, U''(0) = -4*barrier  →  |ω_barrier| = sqrt(4*barrier)
    Barrier height: ΔU = barrier (from well to saddle)

    Kramers rate (overdamped): r = ω_well * |ω_barrier| / (2π) * exp(-ΔU/T)
    """
    if T < 1e-15:
        return 0.0
    omega_well = np.sqrt(8.0 * barrier)
    omega_barrier = np.sqrt(4.0 * barrier)
    delta_U = barrier
    return omega_well * omega_barrier / (2.0 * np.pi) * np.exp(-delta_U / T)


def kramers_lambda1(barrier: float, T: float) -> float:
    """
    λ₁ ≈ 2 * kramers_rate (two escape routes: left→right and right→left).
    """
    return 2.0 * kramers_rate(barrier, T)


def sweep_phase_diagram(barriers: list, temperatures: list, n_pts: int = 401) -> list:
    """
    Compute spectral properties for a grid of (barrier, T) values.
    Returns list of dicts (one per point).
    """
    results = []
    for b in barriers:
        for t in temperatures:
            r = compute_eigenvalues(b, t, n_eigs=6, n_pts=n_pts)
            r_kramers = kramers_lambda1(b, t)
            results.append(
                {
                    "barrier": b,
                    "T": t,
                    "lambda_1": r["lambda_1"],
                    "lambda_2": r["lambda_2"],
                    "gap_ratio": r["gap_ratio"],
                    "tau_1": r["tau_1"],
                    "tau_2": r["tau_2"],
                    "timescale_ratio": r["timescale_ratio"],
                    "kramers_lambda1": r_kramers,
                    "kramers_ratio": r_kramers / r["lambda_1"] if r["lambda_1"] > 1e-15 else np.nan,
                }
            )
    return results
