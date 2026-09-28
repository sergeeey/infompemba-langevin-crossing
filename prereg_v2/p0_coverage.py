"""t=0 only: does P0 (hot farther than cold, KL and W1) hold on the (b,T) grid at kappa=0?
No dynamics are computed here. Design check for the pre-registration."""

import numpy as np

B = [0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
TS = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
x = np.linspace(-10, 10, 20001)
dx = x[1] - x[0]


def norm_log(lp):
    m = lp.max()
    z = m + np.log(np.sum(np.exp(lp - m)) * dx)
    return lp - z


def kl(lp, lq):
    p = np.exp(lp)
    return float(np.sum(p * (lp - lq)) * dx)


def w1(lp, lq):
    return float(np.sum(np.abs(np.cumsum(np.exp(lp)) * dx - np.cumsum(np.exp(lq)) * dx)) * dx)


def families(b, T):
    U = b * (x**2 - 1) ** 2
    out = {}
    for s in (2.0, 3.0):
        out[f"gauss_s{s:g}"] = norm_log(-0.5 * (x / s) ** 2)
    for f in (4, 8):
        out[f"thermal_x{f}T"] = norm_log(-U / (f * T))
    return U, out


rows = []
for b in B:
    for T in TS:
        U, hot = families(b, T)
        leq = norm_log(-U / T)
        # cold = right-well restriction of eq
        lc = np.where(x > 0, leq, -np.inf)
        lc = norm_log(np.where(x > 0, leq, -1e300))
        kc = kl(lc, leq)
        wc = w1(lc, leq)
        for name, lh in hot.items():
            kh, wh = kl(lh, leq), w1(lh, leq)
            rows.append((b, T, name, kc, kh, wc, wh, kh >= 1.25 * kc and wh >= 1.25 * wc))

names = sorted({r[2] for r in rows})
print("P0 (>=1.25x in KL and W1) coverage at kappa=0, of 72 grid points:")
for n in names:
    ok = sum(1 for r in rows if r[2] == n and r[7])
    okk = sum(1 for r in rows if r[2] == n and r[4] >= 1.25 * r[3])
    okw = sum(1 for r in rows if r[2] == n and r[6] >= 1.25 * r[5])
    print(f"  {n:14s} both={ok:2d}  KL-only={okk:2d}  W1-only={okw:2d}")
print("\nsample b=2,T=0.2 :")
for r in rows:
    if r[0] == 2.0 and r[1] == 0.2:
        print(f"  {r[2]:14s} KLc={r[3]:.3f} KLh={r[4]:.3f} W1c={r[5]:.3f} W1h={r[6]:.3f} P0={r[7]}")
