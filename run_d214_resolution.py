"""D2.14 (PREREG_v2.md): resolution (dx, time step) and wall-position (L) checks on 11 configurations.

The plan and the pass criteria A1-A5 were committed (c4e5cf8) before this file was written. This script only
produces raw distance series; prereg_v2/analyze_d214.py applies the criteria.

For each configuration and each task (dx, L) the (cold, hot) pair is evolved at three time-step levels
(geometric growth 1.002, 1.002**(1/2), 1.002**(1/4); record strides 10, 20, 40 keep the recorded times aligned).
Tasks: dx in {0.01, 0.005, 0.0025} at L = 15, and L in {12, 21} at dx = 0.005.
Output: prereg_v2/out_d214/<cfg>_dx<dx>_L<L>.npz (git-ignored), one file per task.
"""

import csv
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.mpemba_exact import (
    cold_ic,
    evolve,
    gaussian_ic,
    make_grid,
    potential,
    time_grid,
)

CONFIGS = [
    (1.0, 0.2, 0.0),
    (2.0, 0.2, 0.0),
    (1.0, 0.1, 0.1),
    (0.1, 1.0, 0.1),
    (3.0, 0.3, 0.05),
    (1.5, 0.05, 0.1),
    (5.0, 0.05, 0.02),
    (5.0, 0.05, 0.05),
    (5.0, 0.05, 0.1),
    (3.0, 0.05, 0.05),
    (5.0, 0.1, 0.05),
]
TASKS = [(0.01, 15.0), (0.005, 15.0), (0.0025, 15.0), (0.005, 12.0), (0.005, 21.0)]
BASE_GROWTH = 1.002
BASE_STRIDE = 10
LEVELS = 3
OUT = Path("prereg_v2/out_d214")


def name(cfg, dx, half_width):
    b, t, k = cfg
    return OUT / f"b{b}_T{t}_k{k}_dx{dx}_L{half_width:g}.npz"


def t_max_of(cfg):
    with open("prereg_v2/summary_main.csv", encoding="utf-8") as fh:
        row = next(
            r
            for r in csv.DictReader(fh)
            if (float(r["b"]), float(r["T"]), float(r["kappa"])) == cfg
        )
    return float(row["t_max"])


def task(args):
    cfg, dx, half_width = args
    b, temp, kappa = cfg
    t0 = time.time()

    def fn(x):
        return potential(x, b, kappa)

    t_max = t_max_of(cfg)  # from the main run's table: same t_max (and lambda1 source) as there
    grid = make_grid(fn, temp, half_width, dx)
    rho0 = np.stack([cold_ic(grid), gaussian_ic(grid, 3.0)], axis=1)
    arrays = {"t_max": np.array(t_max)}
    for j in range(LEVELS):
        traj = evolve(
            grid,
            rho0,
            time_grid(1.05 * t_max, BASE_GROWTH ** (1.0 / 2**j)),
            record_stride=BASE_STRIDE * 2**j,
        )
        arrays[f"t{j}"] = traj.t
        arrays[f"kl{j}"] = traj.kl
        arrays[f"w1{j}"] = traj.w1
        arrays[f"minrho{j}"] = np.array(traj.min_rho)
        arrays[f"drift{j}"] = np.array(traj.mass_drift)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(name(cfg, dx, half_width), **arrays)
    return cfg, dx, half_width, time.time() - t0


def main() -> int:
    smoke = "--smoke" in sys.argv
    jobs = [(c, dx, hw) for c in CONFIGS for dx, hw in TASKS]
    if smoke:
        jobs = [((0.1, 1.0, 0.1), 0.01, 15.0)]
    jobs = [j for j in jobs if not name(*j).exists()]  # resumable
    print(f"{len(jobs)} tasks", flush=True)
    with ProcessPoolExecutor(max_workers=1 if smoke else 22) as pool:
        futures = [pool.submit(task, j) for j in jobs]
        for fut in as_completed(futures):
            cfg, dx, hw, sec = fut.result()
            print(f"done {cfg} dx={dx} L={hw:g} ({sec:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
