"""Apply the D2.14 criteria A1-A5 (PREREG_v2.md, committed before the run) to prereg_v2/out_d214/*.npz.

Run after `python run_d214_resolution.py`. Writes prereg_v2/d214_summary.csv and prints a report;
exit code 0 only if every criterion holds in every configuration. The thresholds are heuristics fixed in advance.
"""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from run_d214_resolution import CONFIGS, OUT, name, t_max_of
from src.mpemba_analysis import D_FLOOR, classify_metric

METRICS = ("kl", "w1")
REL_DEFAULT = 1e-10  # a difference counts only if it exceeds REL * |D|: below that it is round-off
REL = next(
    (float(a.split("=")[1]) for a in sys.argv if a.startswith("--rel=")), REL_DEFAULT
)
DX_FINE, DX_MID, DX_COARSE = 0.0025, 0.005, 0.01


def load(cfg, dx, half_width):
    z = np.load(name(cfg, dx, half_width))
    return {k: z[k] for k in z.files}


def series(z, metric, level, n):
    return z[f"{metric}{level}"][:n]


def common_n(zs):
    n = min(zz[f"t{j}"].size for zz in zs for j in range(3))
    t = zs[0]["t0"][:n]
    for zz in zs:
        for j in range(3):
            assert np.allclose(zz[f"t{j}"][:n], t, rtol=1e-9), (
                "recorded times are misaligned"
            )
    return n, t


def extrap(z, metric, n):
    """First-order Richardson with the two finest time levels (h/2, h/4): 2*fine - coarse."""
    return 2.0 * series(z, metric, 2, n) - series(z, metric, 1, n)


def stats(values):
    return (
        (
            float(np.median(values)),
            float(np.percentile(values, 5)),
            float(np.percentile(values, 95)),
        )
        if values.size
        else (float("nan"),) * 3
    )


def order(delta_coarse, delta_fine, ref, valid):
    ok = valid & (delta_coarse > REL * np.abs(ref)) & (delta_fine > REL * np.abs(ref))
    return stats(np.log2(delta_coarse[ok] / delta_fine[ok])), int(ok.sum())


def main() -> int:
    with open("prereg_v2/summary_main.csv", encoding="utf-8") as fh:
        main_rows = {
            (float(r["b"]), float(r["T"]), float(r["kappa"])): r
            for r in csv.DictReader(fh)
        }
    out_rows, fails = [], {k: [] for k in ("A1", "A2", "A3", "A4", "A5")}
    for cfg in CONFIGS:
        if not all(
            name(cfg, dx, hw).exists()
            for dx, hw in (
                (DX_COARSE, 15.0),
                (DX_MID, 15.0),
                (DX_FINE, 15.0),
                (DX_MID, 12.0),
                (DX_MID, 21.0),
            )
        ):
            print(f"{cfg}: files missing, skipped")
            fails["A1"].append((cfg, "missing files"))
            continue
        t_max = t_max_of(cfg)
        zc, zm, zf = (
            load(cfg, DX_COARSE, 15.0),
            load(cfg, DX_MID, 15.0),
            load(cfg, DX_FINE, 15.0),
        )
        z12, z21 = load(cfg, DX_MID, 12.0), load(cfg, DX_MID, 21.0)
        old = np.load(f"prereg_v2/out/b{cfg[0]}_T{cfg[1]}_k{cfg[2]}.npz")
        n, t = common_n([zc, zm, zf, z12, z21])
        n = min(n, old["t"].size)
        assert np.allclose(old["t"][:n], t[:n], rtol=1e-9), "main-run times differ"
        t = t[:n]
        main_row = main_rows[cfg]
        for m in METRICS:
            ex_m, ex_f = extrap(zm, m, n), extrap(zf, m, n)
            d_ref = ex_f
            err_ref = np.abs(ex_m - ex_f) + np.abs(
                series(zf, m, 2, n) - series(zf, m, 1, n)
            )
            out = classify_metric(t, d_ref, err_ref, t_max)
            sel = (t > 0) & (t <= t_max) & (d_ref[:, 0] >= D_FLOOR)
            i = int(np.flatnonzero(sel)[-1])
            valid = sel & (series(zf, m, 2, n)[:, 0] >= D_FLOOR)
            # A1
            label_main = main_row[f"{m}_label"]
            if out.label != label_main:
                fails["A1"].append((cfg, m, label_main, out.label))
            # A2
            d_old, e_old = old[f"d_{m}"][:n], old[f"err_{m}"][:n]
            diff = np.abs(d_ref[i] - d_old[i])
            a2 = diff / e_old[i]
            if np.any(diff > e_old[i]):
                fails["A2"].append((cfg, m, a2.tolist()))
            gap_ref, gap_old = d_ref[i, 0] - d_ref[i, 1], d_old[i, 0] - d_old[i, 1]
            out_rows.append(
                {
                    "b": cfg[0],
                    "T": cfg[1],
                    "kappa": cfg[2],
                    "metric": m,
                    "state": "gap_vs_main",
                    "gap_rel_change_vs_main": abs(gap_ref - gap_old) / abs(gap_old),
                }
            )
            # A3: observed time order at dx = 0.005 (raw levels h, h/2, h/4)
            raw = [series(zm, m, j, n) for j in range(3)]
            for c, who in ((0, "cold"), (1, "hot")):
                dc, df = (
                    np.abs(raw[0][:, c] - raw[1][:, c]),
                    np.abs(raw[1][:, c] - raw[2][:, c]),
                )
                (pt_med, pt_lo, pt_hi), npt = order(dc, df, raw[2][:, c], valid)
                if not (0.8 <= pt_med <= 1.2):
                    fails["A3"].append((cfg, m, who, pt_med, npt))
                # space order at the finest time level (only reported)
                s = [series(z, m, 2, n)[:, c] for z in (zc, zm, zf)]
                (px_med, _, _), npx = order(
                    np.abs(s[0] - s[1]), np.abs(s[1] - s[2]), s[2], valid
                )
                # A4: interaction of the time difference with dx
                dt_c = np.abs(series(zc, m, 0, n)[:, c] - series(zc, m, 1, n)[:, c])
                dt_f = np.abs(series(zf, m, 0, n)[:, c] - series(zf, m, 1, n)[:, c])
                ok = valid & (dt_f > REL * np.abs(s[2]))
                inter = np.abs(dt_c[ok] - dt_f[ok]) / dt_f[ok]
                inter_med = float(np.median(inter)) if inter.size else float("nan")
                if not inter_med <= 0.10:
                    fails["A4"].append((cfg, m, who, inter_med))
                out_rows.append(
                    {
                        "b": cfg[0],
                        "T": cfg[1],
                        "kappa": cfg[2],
                        "metric": m,
                        "state": who,
                        "p_t_median": pt_med,
                        "p_t_p5": pt_lo,
                        "p_t_p95": pt_hi,
                        "p_t_n": npt,
                        "p_x_median": px_med,
                        "p_x_n": npx,
                        "interaction_median": inter_med,
                        "A2_diff_over_main_err": float(a2[c]),
                        "label_main": label_main,
                        "label_refined": out.label,
                        "p0_main": float(main_row[f"{m}_p0"]),
                        "p0_refined": out.p0_ratio,
                    }
                )
            # A5: wall position at dx = 0.005 (error estimate of the L = 15 refined run)
            gap15 = ex_m[i, 0] - ex_m[i, 1]
            for hw, zz in ((12, z12), (21, z21)):
                dl = extrap(zz, m, n)
                lab = classify_metric(t, dl, err_ref, t_max).label
                gap_l = dl[i, 0] - dl[i, 1]
                rel = abs(gap_l - gap15) / abs(gap15)
                if lab != out.label or rel > 0.01:
                    fails["A5"].append((cfg, m, f"L={hw}", lab, rel))
                out_rows.append(
                    {
                        "b": cfg[0],
                        "T": cfg[1],
                        "kappa": cfg[2],
                        "metric": m,
                        "state": f"gap_L{hw}",
                        "label_refined": lab,
                        "A5_gap_rel_change": rel,
                    }
                )
        sane = min(
            float(zz[f"minrho{j}"]) for zz in (zc, zm, zf, z12, z21) for j in range(3)
        )
        drift = max(
            float(zz[f"drift{j}"]) for zz in (zc, zm, zf, z12, z21) for j in range(3)
        )
        print(f"{cfg}: done (min rho {sane:.2e}, max mass drift {drift:.2e})")
    keys = sorted({k for r in out_rows for k in r})
    csv_path = (
        "prereg_v2/d214_summary.csv"
        if REL == REL_DEFAULT
        else f"prereg_v2/d214_summary_rel{REL:g}.csv"
    )
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(out_rows)
    print()
    for crit, items in fails.items():
        print(f"{crit}: {'PASS' if not items else 'FAIL (' + str(len(items)) + ')'}")
        for it in items[:12]:
            print("   ", it)
    return 0 if not any(fails.values()) else 1


if __name__ == "__main__":
    assert Path("prereg_v2/out_d214").exists() and OUT.exists()
    sys.exit(main())
