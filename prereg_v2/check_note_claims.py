"""Transcription-consistency check between paper/note_v2_draft.md and the saved pipeline outputs.

This is NOT an independent re-derivation: most "recomputed" values below re-read the same cached pipeline
columns (kl_label, r, c1_hot, pl_noise_model, pl_v1pct, v1_sd, ...) the note's numbers were themselves
transcribed from, so it catches transcription/rounding errors and later text drift, not errors shared with
the pipeline itself. Coverage is partial (not every claim in the note has a check here).

Each check prints: claim (as stated in the note), recomputed value, OK/MISMATCH. Inputs: summary_main.csv,
summary_robustness.csv, v1_compare_points.csv (h = 0.025), the four time-step variants of the reproduction
(argument: directory with cmp_sub{5,10,20,40}.csv) and out/*.npz. Read-only. The 'note says' strings were
transcribed from the note / RESULTS.md and grep-checked against them (2026-09-28); only the 'recomputed'
values come from data. A 4th-pass skeptic review (2026-09-29) found the bound checks below used narrow
two-sided windows instead of one-sided ">=" comparisons, which let the note's rounded "1.7"/"1.9" pass
despite the true minima being 1.68/1.88 — fixed below to compare against the note's own stated one-sided
bound directly.
"""

import csv
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.mpemba_analysis import D_FLOOR, MARGIN

RECHECK = sys.argv[1] if len(sys.argv) > 1 else None
results = []


def chk(label, ok, stated, computed):
    results.append(ok)
    print(f"[{'OK' if ok else 'MISMATCH'}] {label}: note says {stated}; recomputed {computed}")


with open("prereg_v2/summary_main.csv", encoding="utf-8") as fh:
    rows = list(csv.DictReader(fh))
for r in rows:
    for k in r:
        if k != "lam_src" and not k.endswith("_label"):
            r[k] = float(r[k])
M = pd.DataFrame(rows)
k0, kn = M[M.kappa == 0], M[M.kappa != 0]

print("=== main run (summary_main.csv) ===")
chk(
    "KL EFFECT",
    (M.kl_label == "EFFECT").sum() == 288,
    "288/288",
    f"{(M.kl_label == 'EFFECT').sum()}/288",
)
chk(
    "W1 EFFECT / NO_TEST",
    (M.w1_label == "EFFECT").sum() == 283 and (M.w1_label == "NO_TEST").sum() == 5,
    "283 / 5",
    f"{(M.w1_label == 'EFFECT').sum()} / {(M.w1_label == 'NO_TEST').sum()}",
)
nt = M[M.w1_label == "NO_TEST"]
chk(
    "NO_TEST all at kappa=0.1, P0 ratio 1.249",
    (nt.kappa == 0.1).all() and nt.w1_p0.between(1.2485, 1.2495).all(),
    "kappa=0.1, 1.249",
    f"kappa {sorted(set(nt.kappa))}, P0 {nt.w1_p0.min():.4f}..{nt.w1_p0.max():.4f}",
)
both = k0[(k0.kl_label != "NO_TEST") & (k0.w1_label != "NO_TEST")]
eb = both[(both.kl_label == "EFFECT") & (both.w1_label == "EFFECT")]
chk(
    "K1: EFFECT_BOTH at kappa=0",
    len(both) == 72 and len(eb) == 72,
    "72 of 72",
    f"{len(eb)} of {len(both)}",
)
kl_only = M[(M.kl_label == "EFFECT") & (M.w1_label != "EFFECT")]
chk(
    "K2: KL-only points = the 5 NO_TEST",
    len(kl_only) == 5 and (kl_only.w1_label == "NO_TEST").all(),
    "5",
    str(len(kl_only)),
)
el = M[np.isfinite(M.r)]
chk(
    "R-eligible points",
    len(el) == 224 and (el.groupby("kappa").size() == 56).all(),
    "224 (56 per kappa)",
    f"{len(el)} ({dict(el.groupby('kappa').size())})",
)
chk("kappa!=0 R-eligible", len(el[el.kappa != 0]) == 168, "168", str(len(el[el.kappa != 0])))
pool = el[(~el.r.between(0.95, 1.05)) & (el.kl_label != "NO_TEST") & np.isfinite(el.kl_lastgap)]
dis = pool[(pool.kl_lastgap > 0) != (pool.r < 1)]
chk(
    "R agreement (excluding [0.95,1.05])",
    len(pool) == 224 and len(dis) == 0,
    "224 of 224",
    f"{len(pool) - len(dis)} of {len(pool)}",
)
rk = el[el.kappa != 0].r
chk(
    "r range at kappa!=0",
    abs(rk.min() - 0.003) < 5e-4 and abs(rk.max() - 0.43) < 5e-3,
    "0.003..0.43",
    f"{rk.min():.4f}..{rk.max():.4f}",
)
e0 = el[el.kappa == 0]
chk(
    "kappa=0 |c1(hot)| max",
    abs(e0.c1_hot.abs().max() - 1.8e-9) < 1e-10,
    "<= 1.8e-9",
    f"{e0.c1_hot.abs().max():.2e}",
)
chk(
    "kappa=0 |c1(cold)| range",
    0.899 < e0.c1_cold.abs().min() < 0.905 and e0.c1_cold.abs().max() < 1.001,
    "0.90..1.00",
    f"{e0.c1_cold.abs().min():.3f}..{e0.c1_cold.abs().max():.3f}",
)
ratio = M.kl_tstar / M.t_max
chk(
    "t*/t_max (KL) range",
    6e-45 < ratio.min() < 7.5e-45 and 1.6e-2 < ratio.max() < 1.8e-2,
    "7e-45..1.7e-2",
    f"{ratio.min():.2e}..{ratio.max():.2e}",
)
chk(
    "lambda1 sources",
    (M.lam_src == "eig").sum() == 224 and (M.lam_src == "quad").sum() == 64,
    "224 eig / 64 quad",
    f"{(M.lam_src == 'eig').sum()} / {(M.lam_src == 'quad').sum()}",
)
chk("KL P0 min ratio", M.kl_p0.min() >= 30.4, ">= 30.4 (exactly 30.4557)", f"{M.kl_p0.min():.4f}")
chk(
    "energy at kappa=0 degenerate",
    (k0.u_p0 > 1e12).all(),
    "72/72 degenerate",
    f"{(k0.u_p0 > 1e12).sum()}/72",
)
chk(
    "energy at kappa!=0 EFFECT",
    (kn.u_label == "EFFECT").sum() == 216,
    "216/216",
    f"{(kn.u_label == 'EFFECT').sum()}/216",
)
chk(
    "Var(x) EFFECT",
    (M.var_label == "EFFECT").sum() == 288,
    "288/288",
    f"{(M.var_label == 'EFFECT').sum()}/288",
)

print("\n=== margins (npz) ===")
med = {"kl": [], "w1": []}
late = {"kl": [], "w1": []}
frac2 = {"kl": [], "w1": []}
at_t = []
for r in rows:
    z = np.load(f"prereg_v2/out/b{r['b']}_T{r['T']}_k{r['kappa']}.npz")
    for m in ("kl", "w1"):
        if r[f"{m}_label"] != "EFFECT":
            continue
        t, d, e = z["t"], z[f"d_{m}"], z[f"err_{m}"]
        sel = (t >= r[f"{m}_tstar"]) & (t <= r["t_max"]) & (d[:, 0] >= D_FLOOR)
        q = ((d[:, 0] - d[:, 1]) / (MARGIN * (e[:, 0] + e[:, 1])))[sel]
        med[m].append(np.median(q))
        late[m].append(q[-1])
        frac2[m].append((q >= 2).mean())
        if m == "kl":
            at_t.append(q[0])
chk(
    "median ratio over window min (KL, W1)",
    min(med["kl"]) >= 164.8 and min(med["w1"]) >= 377,
    ">= 164.8, >= 377",
    f"{min(med['kl']):.1f}, {min(med['w1']):.1f}",
)
chk(
    "fraction of window records with ratio >= 2",
    min(frac2["kl"] + frac2["w1"]) >= 0.995,
    ">= 0.995",
    f"{min(frac2['kl'] + frac2['w1']):.4f}",
)
chk(
    "late-time ratio min (KL, W1)",
    min(late["kl"]) >= 1.68 and min(late["w1"]) >= 1.88,
    ">= 1.68, >= 1.88",
    f"{min(late['kl']):.2f}, {min(late['w1']):.2f}",
)
chk(
    "ratio at t* (KL) min / median",
    1.0 < min(at_t) < 1.002 and 2.2 < np.median(at_t) < 2.3,
    "1.001 / 2.26",
    f"{min(at_t):.3f} / {np.median(at_t):.2f}",
)
cells = ((M["T"] / (8 * M["b"])) ** 0.5 / 0.005).min()
chk("cells per well width at dx=0.005", cells >= 7.0, ">= 7", f"{cells:.2f}")

print("\n=== hot-family sweep (summary_robustness.csv) ===")
S = pd.read_csv("prereg_v2/summary_robustness.csv")
chk(
    "225 runs, P0 in KL",
    len(S) == 225 and (S.kl_label != "NO_TEST").all(),
    "225, all",
    f"{len(S)}, {(S.kl_label != 'NO_TEST').sum()}",
)
kl_e = {s: int(((S.sigma == s) & (S.kl_label == "EFFECT")).sum()) for s in (1.0, 1.5, 3.0)}
chk("KL EFFECT by sigma", kl_e == {1.0: 55, 1.5: 65, 3.0: 75}, "55/65/75", str(kl_e))
nt_s = {s: int(((S.sigma == s) & (S.w1_label == "NO_TEST")).sum()) for s in (1.0, 1.5, 3.0)}
chk("W1 NO_TEST by sigma", nt_s == {1.0: 55, 1.5: 55, 3.0: 5}, "55/55/5", str(nt_s))
mu2 = S[S.mu.abs() == 2]
w = {}
for s in (1.0, 1.5, 3.0):
    sub = mu2[(mu2.sigma == s) & (mu2.w1_label != "NO_TEST")]
    w[s] = (int((sub.w1_label == "EFFECT").sum()), len(sub))
chk(
    "W1 EFFECT / testable at |mu|=2 (unmatched; no longer quoted in the note)",
    w == {1.0: (0, 20), 1.5: (10, 20), 3.0: (30, 30)},
    "0/20, 10/20, 30/30",
    str(w),
)
key = ["b", "T", "kappa", "mu"]
piv = mu2.pivot_table(index=key, columns="sigma", values="w1_label", aggfunc="first")
common = piv[piv.apply(lambda r: all(v != "NO_TEST" for v in r), axis=1)]
cw = {sg: int((common[sg] == "EFFECT").sum()) for sg in (1.0, 1.5, 3.0)}
chk(
    "common-support W1 comparison at |mu|=2",
    len(common) == 20 and cw == {1.0: 0, 1.5: 10, 3.0: 20},
    "20 records; 0, 10, 20",
    f"{len(common)}; {cw}",
)
cs = common.reset_index()
cs["a"] = (cs["b"] / cs["T"]).round(9)
n_cond = cs[["a", "kappa", "mu"]].drop_duplicates().shape[0]
chk("distinct dimensionless conditions in the common set", n_cond == 4, "4", str(n_cond))
extra3 = mu2[mu2.sigma == 3.0].set_index(key).drop(index=common.index, errors="ignore")
chk(
    "sigma=3 extra records all EFFECT",
    len(extra3) == 10 and (extra3.w1_label == "EFFECT").all(),
    "10, all EFFECT",
    f"{len(extra3)}, {extra3.w1_label.value_counts().to_dict()}",
)
sig1 = mu2[(mu2.sigma == 1.0) & (mu2.w1_label != "NO_TEST")]
chk(
    "testable sigma=1 W1 records are all CROSSING_NO_MARGIN",
    len(sig1) == 20 and (sig1.w1_label == "CROSSING_NO_MARGIN").all(),
    "20, all CROSSING_NO_MARGIN",
    str(sig1.w1_label.value_counts().to_dict()),
)
n_var = S[["kappa", "mu", "sigma"]].drop_duplicates().shape[0]
n_ratio = (S["b"] / S["T"]).round(9).nunique()
chk(
    "sweep: 45 distinct (kappa, mu, sigma), one b/T",
    n_var == 45 and n_ratio == 1,
    "45; 1",
    f"{n_var}; {n_ratio}",
)
ma = (M["b"] / M["T"]).round(9)
n_pairs = M[["b", "T"]].drop_duplicates().shape[0]
n_cfg = pd.concat([ma, M.kappa], axis=1).drop_duplicates().shape[0]
chk(
    "main run: 72 (b,T) pairs, 46 distinct b/T, 184 distinct (b/T, kappa)",
    n_pairs == 72 and ma.nunique() == 46 and n_cfg == 184,
    "72 / 46 / 184",
    f"{n_pairs} / {ma.nunique()} / {n_cfg}",
)
narrow_test = S[(S.sigma < 3) & (S.w1_label != "NO_TEST")]
chk(
    "narrow states testable only at |mu|=2",
    (narrow_test.mu.abs() == 2).all(),
    "only |mu|=2",
    str(sorted(set(narrow_test.mu))),
)
nm = S[S.kl_label == "CROSSING_NO_MARGIN"]
chk(
    "30 no-margin runs, all |mu|=2, 20 with |r-1|<=0.05, r 0.87..0.97",
    len(nm) == 30
    and (nm.mu.abs() == 2).all()
    and int((abs(nm.r - 1) <= 0.05).sum()) == 20
    and 0.87 <= nm.r.min()
    and nm.r.max() <= 0.972,
    "30 / |mu|=2 / 20 / 0.87..0.97",
    f"{len(nm)} / {sorted(set(nm.mu.abs()))} / {int((abs(nm.r - 1) <= 0.05).sum())} / {nm.r.min():.3f}..{nm.r.max():.3f}",
)
split = (int((nm.sigma == 1.0).sum()), int((nm.sigma == 1.5).sum()))
chk("no-margin split by sigma", split == (20, 10), "20 (sigma=1), 10 (sigma=1.5)", str(split))
agree = ((S.kl_lastgap > 0) == (S.r < 1)).sum()
chk(
    "R agreement in sweep, r<1 everywhere",
    agree == 225 and (S.r < 1).all(),
    "225/225, r<1 all",
    f"{agree}/225, r<1: {(S.r < 1).sum()}",
)

print("\n=== v1 comparison (v1_compare_points.csv, h = 0.025) ===")
C = pd.read_csv("prereg_v2/v1_compare_points.csv")
pear = np.corrcoef(C.pl_v1pct, C.v1_mean)[0, 1]
mad = (C.pl_v1pct - C.v1_mean).abs().mean()
chk(
    "Pearson / mean discrepancy (h=0.025)",
    abs(pear - 0.33) < 0.005 and abs(mad - 24.7) < 0.1,
    "0.33 / 24.7",
    f"{pear:.3f} / {mad:.2f}",
)
chk(
    "points > 70%: noise-free / v1",
    int((C.pl_v1pct > 70).sum()) == 57 and int((C.v1_mean > 70).sum()) == 27,
    "57 / 27",
    f"{int((C.pl_v1pct > 70).sum())} / {int((C.v1_mean > 70).sum())}",
)
deg = C.pl_max_abs_gap < 1e-6
chk("degenerate points", int(deg.sum()) == 11, "11", str(int(deg.sum())))
chk(
    "earlier sd: degenerate / other (medians)",
    51 < C.v1_sd[deg].median() < 52 and 4.5 < C.v1_sd[~deg].median() < 5,
    "51.4 / 4.7",
    f"{C.v1_sd[deg].median():.1f} / {C.v1_sd[~deg].median():.1f}",
)
chk(
    "mean discrepancy at non-degenerate",
    abs(C.abs_diff[~deg].mean() - 19.8) < 0.1,
    "19.8",
    f"{C.abs_diff[~deg].mean():.1f}",
)
f = C.pl_frac_below_theta
preds = {
    "grand": pd.Series(C.v1_mean.mean(), index=C.index),
    "hybrid": pd.Series(np.where(deg, 50.0, C.pl_v1pct), index=C.index),
    "shrink": (1 - f) * C.pl_v1pct + 50 * f,
    "model": C.pl_noise_model,
}
err = {k: (v - C.v1_mean).abs() for k, v in preds.items()}
chk(
    "baselines 18.5 / 19.1 / 9.9 and model 8.8",
    [round(err[k].mean(), 1) for k in ("grand", "hybrid", "shrink", "model")]
    == [18.5, 19.1, 9.9, 8.8],
    "18.5 / 19.1 / 9.9 / 8.8",
    str([round(err[k].mean(), 1) for k in ("grand", "hybrid", "shrink", "model")]),
)
chk(
    "stratified: degenerate 15.4; other model 7.6, shrink 8.9",
    round(err["model"][deg].mean(), 1) == 15.4
    and round(err["model"][~deg].mean(), 1) == 7.6
    and round(err["shrink"][~deg].mean(), 1) == 8.9,
    "15.4 / 7.6 / 8.9",
    f"{err['model'][deg].mean():.1f} / {err['model'][~deg].mean():.1f} / {err['shrink'][~deg].mean():.1f}",
)
se = C.v1_sd / np.sqrt(10)
chk(
    "noise floors: all 3.5, degenerate 12.3, other 1.9",
    [round(0.8 * se.mean(), 1), round(0.8 * se[deg].mean(), 1), round(0.8 * se[~deg].mean(), 1)]
    == [3.5, 12.3, 1.9],
    "3.5 / 12.3 / 1.9",
    str(
        [round(0.8 * se.mean(), 1), round(0.8 * se[deg].mean(), 1), round(0.8 * se[~deg].mean(), 1)]
    ),
)
sw = [
    (C[c] - C.v1_mean).abs().mean()
    for c in (
        "pl_noise_model_t0.005",
        "pl_noise_model_t0.01",
        "pl_noise_model",
        "pl_noise_model_t0.03",
    )
]
chk(
    "theta 0.005..0.03 range 6.9..9.8",
    abs(min(sw) - 6.9) < 0.06 and abs(max(sw) - 9.8) < 0.06,
    "6.9..9.8",
    f"{min(sw):.2f}..{max(sw):.2f}",
)
chk(
    "57 points with majority below theta; overlap 42",
    int((f > 0.5).sum()) == 57 and len(set(C.index[C.pl_v1pct > 70]) & set(C.index[f > 0.5])) == 42,
    "57 / 42",
    f"{int((f > 0.5).sum())} / {len(set(C.index[C.pl_v1pct > 70]) & set(C.index[f > 0.5]))}",
)
rnd = C.pl_max_abs_gap < 1e-13
alt = np.where(rnd, 50.0, C.pl_v1pct)
chk(
    "round-off points 4; Pearson 0.39, mean 22.6, 56 points",
    int(rnd.sum()) == 4
    and round(np.corrcoef(alt, C.v1_mean)[0, 1], 2) == 0.39
    and round(np.abs(alt - C.v1_mean).mean(), 1) == 22.6
    and int((alt > 70).sum()) == 56,
    "4 / 0.39 / 22.6 / 56",
    f"{int(rnd.sum())} / {np.corrcoef(alt, C.v1_mean)[0, 1]:.2f} / {np.abs(alt - C.v1_mean).mean():.1f} / {int((alt > 70).sum())}",
)
ref = C[(C.b == 1.0) & (C["T"] == 0.2)].iloc[0]
chk(
    "reference point noise-free 95.0 vs 94.8+-0.6",
    abs(ref.pl_v1pct - 95.0) < 0.05
    and abs(ref.v1_mean - 94.8) < 0.05
    and abs(ref.v1_sd - 0.6) < 0.05,
    "95.0 vs 94.8 +- 0.6",
    f"{ref.pl_v1pct:.1f} vs {ref.v1_mean:.1f} +- {ref.v1_sd:.1f}",
)

if RECHECK:
    print("\n=== time-step variants (four runs of the reproduction) ===")
    stats = {}
    for sub in (5, 10, 20, 40):
        D = pd.read_csv(f"{RECHECK}/cmp_sub{sub}.csv")
        stats[sub] = {
            "pearson": np.corrcoef(D.pl_v1pct, D.v1_mean)[0, 1],
            "mad": (D.pl_v1pct - D.v1_mean).abs().mean(),
            "a_n": int((D.pl_v1pct > 70).sum()),
            "a_mean": D.pl_v1pct.mean(),
            "b_n": int((D.pl_analytic_eq_pct > 70).sum()),
            "b_mean": D.pl_analytic_eq_pct.mean(),
            "b_min": D.pl_analytic_eq_pct.min(),
            "first": int((D.pl_first == 0).sum()),
        }
    P = [stats[s] for s in (5, 10, 20, 40)]
    chk(
        "Pearson x4",
        [round(p["pearson"], 2) for p in P] == [0.25, 0.33, 0.22, 0.29],
        "0.25/0.33/0.22/0.29",
        str([round(p["pearson"], 2) for p in P]),
    )
    chk(
        "mean discrepancy x4",
        [round(p["mad"], 1) for p in P] == [25.4, 24.7, 26.2, 24.9],
        "25.4/24.7/26.2/24.9",
        str([round(p["mad"], 1) for p in P]),
    )
    chk(
        "ablation A points x4",
        [p["a_n"] for p in P] == [62, 57, 61, 59],
        "62/57/61/59",
        str([p["a_n"] for p in P]),
    )
    chk(
        "ablation A mean x4",
        [round(p["a_mean"], 1) for p in P] == [88.3, 82.4, 86.0, 83.5],
        "88.3/82.4/86.0/83.5",
        str([round(p["a_mean"], 1) for p in P]),
    )
    chk(
        "ablation B points x4",
        [p["b_n"] for p in P] == [72, 72, 70, 72],
        "72/72/70/72",
        str([p["b_n"] for p in P]),
    )
    chk(
        "ablation B mean x4",
        [round(p["b_mean"], 1) for p in P] == [99.9, 99.8, 98.1, 100.0],
        "99.9/99.8/98.1/100.0",
        str([round(p["b_mean"], 1) for p in P]),
    )
    chk(
        "ablation B minimum x4",
        [round(p["b_min"], 1) for p in P] == [99.2, 88.0, 25.5, 99.3],
        "99.2/88.0/25.5/99.3",
        str([round(p["b_min"], 1) for p in P]),
    )
    chk(
        "first record gap<0 x4",
        [p["first"] for p in P] == [71, 70, 70, 69],
        "71/70/70/69",
        str([p["first"] for p in P]),
    )

print(f"\n{sum(results)} of {len(results)} claims reproduced")
sys.exit(0 if all(results) else 1)
