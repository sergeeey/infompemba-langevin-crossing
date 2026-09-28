"""Where does |c1(hot)| ~ 1e-9 at kappa = 0 come from? Read-only diagnostic (skeptic item 1.8).

At kappa = 0 the symmetric hot state has c1 = 0 exactly in exact arithmetic. Here we measure how
far the computed slowest eigenvector phi1 is from being odd, and how c1(hot) scales with the
eigenvector error estimate eps * ||S|| / lambda1 (mixing with the zero mode).
"""

import sys

import numpy as np
from scipy.linalg import eigh_tridiagonal

sys.path.insert(0, ".")
from src.mpemba_analysis import slow_mode_overlaps
from src.mpemba_exact import gaussian_ic, make_grid, out_rates, potential

for b, T in [(2.0, 0.2), (3.0, 0.3), (5.0, 0.5), (0.5, 0.05)]:
    for dx in (0.01, 0.005):
        g = make_grid(lambda x, b=b: potential(x, b, 0.0), T, 15.0, dx)
        n = g.x.size
        diag, off = -out_rates(g), np.sqrt(g.up * g.dn)
        w, v = eigh_tridiagonal(diag, off, select="i", select_range=(n - 2, n - 2))
        phi = v[:, 0]
        odd_err = float(np.abs(phi + phi[::-1]).max() / np.abs(phi).max())
        lam, (c_h,) = slow_mode_overlaps(g, [gaussian_ic(g, 3.0)])
        bound = 2.2e-16 * float(np.abs(diag).max()) / lam
        print(
            f"b={b} T={T} dx={dx}: lambda1={lam:.2e}  max|phi1(x)+phi1(-x)|/max|phi1| = {odd_err:.1e}  "
            f"|c1(hot)| = {abs(c_h):.1e}  eps*||S||/lambda1 = {bound:.1e}"
        )
