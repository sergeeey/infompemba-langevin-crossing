# A Mpemba "crossing" that the metric could not fail to report: an audit of an earlier analysis and a pre-specified re-analysis of a double-well Langevin model

**DRAFT v0.2 — not for submission.** Bracketed items `[...]` are decisions or facts only the author can supply.

**Author:** [NAME — the files of the earlier draft say "Sergei Boiko", the git configuration says "Sergey Boyko"; to be settled by the author]
**Affiliation:** [AFFILIATION] **Correspondence:** [E-MAIL]
**AI-assistance disclosure:** [text required by the target venue. The analysis code, the numerical validation and the drafting of this note were done with an AI coding assistant (Claude); the venue's policy has not been checked.]
**Status of the earlier draft:** [AUTHOR TO STATE: whether the earlier draft was ever posted or circulated, and where. If it was public, this note should be a replacement or withdrawal with a cross-link.]

## Abstract

An earlier draft of this work reported a "Mpemba-like crossing" in the basin-occupancy observable of an overdamped Langevin particle in a double-well potential
(94.7% of the recorded time at its reference point) and argued that the effect is observable-specific. A pre-publication audit showed that the analysis could
not have detected the effect it claimed. Here we document why and re-examine the model with a numerically converged solution of the one-dimensional
Fokker–Planck equation and a pre-specified protocol with controls. (1) On noise-free trajectories, dropping the check that the hot state starts farther from
equilibrium is by itself sufficient to produce a "crossing" at 72 of 72 grid points (mean 99.8% of the time) even with the analytic equilibrium; the check alone removes
it entirely (0 of 72). Estimating the equilibrium from the tails of the compared trajectories does not create the artefact but changes its numbers. (2) The earlier
phase map is not reproduced by that metric on noise-free trajectories (Pearson r = 0.33); a post-hoc model in which every record whose true gap lies below the
sampling noise is a coin flip reduces the mean discrepancy from 24.7 to 6.9–9.8 percentage points, i.e. most of the map is the sign of noise. (3) With the flaws
removed, a symmetric hot state crosses a one-well-restricted cold state in the Kullback–Leibler distance at all 288 parameter combinations, and in both Kullback–Leibler
and Wasserstein-1 at 283 of them (at the other five the Wasserstein-1 precondition is missed by a heuristic threshold); at κ = 0 this is guaranteed by symmetry
(the hot state has no overlap with the odd slowest mode). The crossing is consistent with the known spectral criterion of Lu and Raz in all 224 points where it was
evaluated, and the conclusion depends strongly on the width of the hot state: in the Wasserstein-1 metric it holds for 70 of 75 runs with a wide hot state and for none
with a narrow one. The mechanism is not new, and closely related results exist; the contribution here is methodological.

## 1. Introduction

The Mpemba effect, in the Markovian setting, is the statement that a system prepared farther from equilibrium can relax faster than one prepared closer to it
[Lu & Raz 2017; Klich et al. 2019]. Its spectral origin — the coefficient of the slowest relaxation mode in the initial state — is established [Lu & Raz 2017;
Klich et al. 2019]; it has been observed for a colloidal particle in a double-well potential [Kumar & Bechhoefer 2020]; and a formulation independent of the
distance measure exists [Vu & Hayakawa 2025]. Closely related to the present model are: an exact spectral treatment of a piecewise-linear double well, which concludes
that neither metastability nor asymmetry of the landscape is necessary or sufficient and that the initial population statistics and the observable matter
[Biswas, Rajesh & Pal 2023]; the dependence of the effect on the distance measure in granular gases [Biswas, Prasad & Rajesh 2023]; and an analytically solvable
two-dimensional bistable model with Kullback–Leibler crossing conditions [Hayakawa & Takada 2026]. See [Teza et al. 2025] for a review.

We do not claim a new mechanism. We report what happened when an analysis of this kind was carried out with a metric that could not fail, how that was found and
what a corrected, validated analysis shows. The earlier analysis is used as a case study; we have one case and make no claim about how common its defects are.

## 2. Model, numerical solution and validation

**Model.** Overdamped Langevin dynamics dx = −U′dt + √(2T)dW with U(x, y) = b(x²−1)² + κ·b·x + 0.3y². The y coordinate is an independent Ornstein–Uhlenbeck
process, so all distances used here are distances of the x marginal and the initial y distribution does not enter. Parameters: b ∈ {0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5},
T ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1}, κ ∈ {0, 0.02, 0.05, 0.1} (288 combinations, b/T from 0.1 to 100). The x grid extends to |x| = 15 with reflecting walls
(the mass of N(0, 3²) beyond |x| = 15 is 6·10⁻⁷; the grid state is the normalised discretised Gaussian).

**Initial states.** Cold: the Boltzmann distribution restricted to x > 0. Hot: a Gaussian N(0, 3²) in x in the main run. The hot state was fixed by a rule that uses
distances at t = 0 only and was applied before any dynamics were computed (Section 3, rule H).

**Numerical solution.** A Scharfetter–Gummel finite-volume generator [Scharfetter & Gummel 1969] with implicit Euler steps on a geometric time grid. The tridiagonal
solve is rewritten so that every update adds positive numbers only, in the spirit of subtraction-free (GTH-type) elimination [Grassmann, Taksar & Heyman 1985]; this
makes the density non-negative by construction and, in our tests, preserved the slow modes at step sizes where ordinary banded elimination did not. We have no error
bound. Two grid steps (dx = 0.01, 0.005) and two time-step ratios give Richardson-extrapolated distances and an error estimate; the error of the finer result is
therefore an estimate, not a proof, and the spatial error at dx = 0.01 is about 1% in KL. A spectral solver was rejected because projecting a wide hot state onto modes
needs exp(U/2T), which exceeds double precision in the tails (exponent about 1.4·10⁴ for b = 5, T = 0.05, |x| = 4.2).

**Validation (tolerances written before the code; failed early runs and the changes they caused are recorded in the repository, `PREREG_v2.md` D1.1–D1.4).** Against the
exact Ornstein–Uhlenbeck solution: relative error ≤ 7.7·10⁻⁴ in KL and W1 at all 1163 recorded times (tolerance 5·10⁻³). Stationarity, mass conservation and positivity
at all 17 290 steps of a test run: residuals 4.3·10⁻¹⁴ and 4.0·10⁻¹⁴, density exactly non-negative. The slowest rate λ₁ from first-passage quadrature agrees with the
symmetrised-generator eigenvalue within a factor 1.079 at all 224 grid points with b/T ≤ 12. An independent Euler–Maruyama simulation (10⁵ particles, three fixed
seed bases, three points at κ = 0) reproduces λ₁ within 7.9% (tolerance 10%) and the Wasserstein-1 curves within 0.0105 (tolerance 0.02). Three gates failed on
first runs, for four reasons (first-order time error; ill-conditioning of banded elimination at large steps; a cancellation floor of 10⁻¹⁷ in the KL evaluation; Euler error in the hot
tails of the cross-check simulation); each was fixed in the method with the tolerance unchanged. Two margins are thin: 1.3× for the simulation cross-check of λ₁ and
1.05× for the grid refinement in KL.

## 3. Pre-specified protocol

Criteria were committed to local version control before the corresponding code and runs (Appendix). The commits are on one machine and have not been pushed to an
external timestamping service, so the ordering cannot at present be verified by a third party.

**P0 (a heuristic filter, not a theorem).** In a given metric the hot state must start at least 1.25 times as far from equilibrium as the cold state. A point failing P0 is
labelled NO_TEST, never "no effect". The value 1.25 is a preset heuristic without calibration.

**EFFECT.** With gap = D_cold − D_hot: the gap must exceed five times the summed discretisation errors at every recorded time from a moment t* on, with t* < t_max/1.05,
t_max = 8/λ₁, excluding times where D_cold < 10⁻¹⁸. The headline "EFFECT_BOTH" requires this in both KL and W1, so that a crossing produced only by the tail sensitivity of KL is
not counted.

**Rule H (hot state).** Candidates in order: Gaussian σ = 2 (the earlier draft's), Gaussian σ = 3, thermal states at 4T and 8T. The first with P0 in both metrics at
≥ 36 of the 72 points at κ = 0 is the main hot state. Pre-freeze check at t = 0: σ = 2 → 0/72 (it is closer than the cold state in W1 at every point), σ = 3 → 72/72, thermal → 0/72.

**Controls.** G3 (positive, 10/10 detected): symmetric hot states at b/T = 10, κ = 0 — a subset of the main design, so it tests detection by the pipeline, not the physics. G4 (negative)
for KL: the pair (cold evolved to s*, cold at t = 0) cannot cross by the data-processing inequality, because the implicit-Euler step matrices are functions of the same generator and
therefore commute, so A_t = M(s*)B_t with M a stochastic map preserving π (8/8 pairs, maximum KL gap −3·10⁻⁹…−3·10⁻⁸; applied to raw runs, since the guarantee does not cover
Richardson-extrapolated values; the pairs satisfy P0 in KL by construction). G4 for W1 uses a convex potential U = x²/2, where the flow is a W1 contraction (3/3 with no EFFECT;
maximum raw W1 gap −7.6·10⁻⁵…−7.9·10⁻⁵). In the double-well G4 pairs the W1 gaps were also negative (−1.4·10⁻⁵…−3.4·10⁻⁵) but no theorem covers W1 there. The originally planned
negative control ("the same state but wider inside the same well") was replaced before running because it is not guaranteed negative: at κ = 0 the cold state has ρ/π − 1 =
sign(x), and a wider state in the same well puts more weight near the barrier where the left eigenfunction is smaller, so its overlap with the slow mode can be smaller and a
crossing is then legitimate.

**Spectral criterion R.** Let g₁ = φ₁/√π be the left eigenvector of the slowest mode (φ₁ the eigenvector of the symmetrised generator) and c₁(ρ) = Σᵢ g₁ᵢρᵢ. R predicts that the hot
state is asymptotically closer iff |c₁(hot)| < |c₁(cold)|. Where π < 10⁻²⁰ the ratio φ₁/√π is round-off dominated and g₁ is continued by a constant from the nearest node with
π ≥ 10⁻²⁰ (g₁ is flat in the drift-dominated tails). For the wide hot state this region carries more than 30% of the hot mass in a test case, so the continuation matters; the late-time
KL amplitude ½c₁²e^(−2λ₁t) obtained from it agrees with the time-stepped solution within 5% (κ = 0.05; a consistency check of the implementation, since both sides use the same discrete
generator). R was evaluated only where λ₁ is resolvable by an eigenvalue computation and the region π ≥ 10⁻²⁰ is contiguous, i.e. b/T ≤ 12 (224 of 288 points, "eligible").

**Decision rules (fixed in advance).** K0: any validation or control gate fails ⇒ stop, no verdict. K1: fewer than 10 EFFECT_BOTH among the P0-eligible κ = 0 points ⇒ reject (not applied
if fewer than 36 of 72 points satisfy P0: design inadequate). K2: EFFECT in KL but nowhere in both metrics ⇒ metric artefact. K3: R agrees with the direct late-time sign in ≥ 95% of eligible
points (excluding r ∈ [0.95, 1.05]) and K1, K2 pass ⇒ the effect is an instance of the known criterion and is not presented as new; K3′: ≥ 3 disagreements after all gates ⇒ bug search, then independent
review. K4: no statement about neural networks may follow from this model. N1: literature check before any wording of novelty. All thresholds are heuristics fixed before the data.

**Post-hoc additions (labelled where they occur).** A post-hoc power check of R with pairs where R predicts no crossing; a sweep over hot states; the exploratory attribution and noise-floor
analyses of Section 5. Definitions of these were committed before their code where stated in the repository (D2.7–D2.9); the noise-floor model was formulated after seeing that the exact
reproduction failed.

## 4. Results

**Main run.** KL: EFFECT at 288 of 288 points. W1: EFFECT at 283, NO_TEST at 5 (all at κ = 0.1, P0 ratios of 1.249 against the threshold 1.25). K1 passed (72 of 72 at κ = 0), K2
passed (the five "KL-only" points are those NO_TEST points), G3/G4/G5 passed. The crossing occurs early, t*/t_max from 7·10⁻⁴⁵ to 1.7·10⁻² (KL, median 5·10⁻⁴); in deep wells it is fast
intra-well relaxation against metastable population imbalance. The smallest margin over the discretisation error is 1.68 (gap/(5·error)) at b/T = 100.

**The κ = 0 result is guaranteed by symmetry.** The symmetric hot state has |c₁| ≤ 1.8·10⁻⁹ while the cold state has |c₁| between 0.90 and 1.00. The residual 10⁻⁹ is not physics:
the computed φ₁ departs from exact oddness by 7·10⁻¹⁰–3.6·10⁻⁹ and |c₁(hot)| tracks half of that, three orders below the a-priori eigenvector error bound ε‖S‖/λ₁ ≈ 4–8·10⁻⁶. The 72
points at κ = 0 therefore record one fact 72 times: they check the implementation, and finding the effect at every one of them is not evidence about the model. With tilt the hot state is
no longer orthogonal to the slow mode, and r = |c₁(hot)|/|c₁(cold)| lies between 0.003 and 0.43. There is no (b, T) region without the effect among the P0-eligible points **for this hot state
on this grid**; this follows from symmetry and continuity and is a property of the design, and it includes shallow barriers (b = 0.1, T = 1) where there is no metastability, consistent with
[Biswas, Rajesh & Pal 2023].

**Consistency of R with the direct calculation.** R and the direct late-time sign agreed in 224 of 224 eligible points. This is a consistency check, not confirmation of the criterion: both
come from the same discrete generator, the asymptotic ordering is fixed by the slowest-mode coefficient, and r < 1 at every eligible point, so it could not have failed in the other direction.
A post-hoc check with pairs built so that R predicts no crossing (r from 5.6 to 10⁹) agreed in 8 of 8.

**Sensitivity to the hot state (post-hoc sweep, 225 runs: 5 contexts at b/T = 10 × κ ∈ {0, 0.05, 0.1} × 15 Gaussian hot states, μ ∈ {−2, −1, 0, 1, 2}, σ ∈ {1, 1.5, 3}).** P0 in KL holds in all 225 runs.
EFFECT in KL: 55/75 (σ = 1), 65/75 (σ = 1.5), 75/75 (σ = 3); the remainder are crossings without margin (hot states in one well with small width, where r approaches 1). EFFECT in W1: 0/75, 10/75, 70/75,
with W1 NO_TEST in 55, 55 and 5 runs. So the "both metrics" headline is a property of a wide hot state; narrow hot states are not farther than the cold state in W1. R and the direct sign agreed in
225 of 225 runs without excluding r ∈ [0.95, 1.05] (20 runs have |r − 1| ≤ 0.05); r < 1 in all 225 (none had r > 1), so even this sweep does not probe the region where R predicts no crossing.

**Other observables (descriptive).** At κ = 0 the cold state sits exactly at the equilibrium value of ⟨U⟩ (its distance is 3·10⁻¹⁷, round-off), so the comparison for energy is degenerate and a statement of
the form "energy shows no crossing" cannot be tested there. At κ ≠ 0, ⟨U⟩ crosses at 216 of 216 points and Var(x) at 288 of 288.

## 5. Case study: the earlier analysis

**The earlier metric.** For each observable the earlier analysis took O_eq as the mean of the last 10% of the two compared trajectories, computed the gap D_hot − D_cold along the run and reported the
fraction of the recorded time with gap < 0 ("% crossed"). It did not check that the hot state started farther. For p_left at κ = 0 the equilibrium is 0.5, the distance is at most 0.5, and a cold state
fully in one well starts at 0.5, so P0 is unsatisfiable (0 of 72 points); with tilt this specific argument no longer holds.

**Ablation on noise-free trajectories (72 points, κ = 0, the earlier initial states):**

| Variant | Points with "crossing" > 70% of the time | Mean |
|---|---|---|
| earlier metric (tail-estimated O_eq, no P0 check) | 57 / 72 | 82.4% |
| analytic O_eq = 0.5, still no P0 check | 72 / 72 | 99.8% (minimum 88.0%) |
| either estimate + P0 check | P0 satisfied at 0 / 72 (NO_TEST) | — |

The missing initial-ordering check is sufficient; the tail-estimated equilibrium is not necessary and lowers the fraction. Its own bias is real: mean deviation 0.095 (maximum 0.25) from 0.5 on exact
trajectories, consistent with 0.097 and 0.259 in the original simulations. The first "crossing" occurred at the first record in 70 of 72 points (646 of 720 runs originally).

**The earlier phase map is not reproduced by its own metric on exact dynamics** (Pearson r = 0.33 between the metric on noise-free trajectories and the earlier ten-seed means; 57 versus 27 points above 70%).
Attribution tests, defined before they were coded: (i) *finite sample and gradient clipping* — rerunning the earlier process (Euler–Maruyama, η = 0.005, 30 000 steps) at four points chosen by a preset rule, with
N = 5000 and N = 50 000, with and without clipping, supported finite sample at 1 of 4 points and clipping at 0 of 4 (threshold 3 of 4): unexplained by either. (ii) *Degeneracy* — with O_eq from the two tails, the
gap is identically zero whenever both trajectories are stationary in the last-10% window (cold stuck near 0, hot at 0.5); the metric then returns the sign of noise. Predictions fixed before the check: at
11 degenerate points (exact max|gap| < 10⁻⁶) the earlier ten-seed standard deviation is 51.4 percentage points against 4.7 elsewhere, and 100% of the individual earlier runs at those points are at an extreme
(< 10% or > 90%) — both confirmed; the prediction that reproduction would be good away from the degenerate set failed (mean discrepancy there 19.8 pp). (iii) *Post-hoc noise-floor model (θ = 0.02 from sampling theory,
formulated after (ii) failed)* — records with |gap| ≤ θ are coin flips, the others deterministic; it reduces the mean discrepancy with the earlier map from 24.7 to 8.8 pp (r = 0.71) and, over θ from 0.005 to 0.03, to
6.9–9.8 pp (r up to 0.82). We conclude that most of the earlier map records the sign of quantities below the sampling noise, without claiming the model is a complete explanation (14 of 72 points remain more than 10 pp off).

**Other unsupported claims of the earlier draft.** (a) "No crossing in energy or position variance": at κ = 0 the energy comparison is degenerate (Section 4); the earlier draft's ⟨x²⟩ and the Var(x) used here are different observables.
(b) A negative result for neural networks was presented as consistent with the mechanism; those experiments were invalid (the "cold" state was not stationary), so the result is "not tested". This note makes no statement about neural networks.
(c) The controls of the earlier draft (balanced cold state, single-well potential, no clipping, 30 seeds) each behaved as the mechanism predicted, but none tested whether the pipeline reports a crossing when the hot state simply starts closer.

## 6. Limitations

One-dimensional reduction; two families of hot and cold states; a single potential family; Markovian overdamped dynamics; KL and W1 as headline metrics. Thresholds (1.25, 5×, 10/72, 95%, 10⁻¹⁸, t_max = 8/λ₁, the factor 1.05) are
preset heuristics; P0 was missed by 0.001 at five points. Margins are thin in the extreme corner (1.68 at b/T = 100) and the spatial error at dx = 0.01 is about 1% in KL; two grid levels cannot confirm the convergence order. For
b/T > 12 (64 points) λ₁ is a first-passage estimate validated only where an eigenvalue computation is possible, and R was not evaluated. The cross-check against simulation used three points at κ = 0. The agreement of R with the
direct calculation is a consistency check. The pre-specification is local. The code was reviewed by an AI reviewer reading the code (final verdict on the solver and on the analysis code: LGTM after fixes; the reviewer could not run
its own probes); there is no independent human replication.

## 7. Conclusion

In this model a symmetric hot state crosses a one-well cold state wherever the precondition holds, because it does not excite the odd slowest mode; this is the known spectral mechanism and is not new. The earlier reported
"phase diagram" and "observable-specificity" were produced by a metric whose central defect — no check that the hot state starts farther — is sufficient by itself to yield a crossing, and whose output at most points is the sign
of noise. Practical checks that follow from this one case: verify the initial ordering (P0); use an analytic or independently converged equilibrium; ask whether the observable's distance to equilibrium can exceed the cold state's initial
distance; test for degeneracy when the cold state already sits at the observable's equilibrium value; and include a control the metric must fail.

## Data and code availability

[Repository URL and archive DOI to be supplied by the author. At present the repository is local only: the pre-specification tag `prereg-v2-frozen` and all commits have not been pushed to any remote.] Per-configuration summary:
`prereg_v2/summary_main.csv`; manifest with SHA-256 of the 288 curve files: `prereg_v2/out_manifest.csv`; sweep: `prereg_v2/summary_robustness.csv`.

## References (identifiers verified in OpenAlex/Crossref; see `prereg_v2/N1_LITERATURE.md`)

- Z. Lu and O. Raz, Proc. Natl. Acad. Sci. USA 114, 5083 (2017), doi:10.1073/pnas.1701264114.
- I. Klich, O. Raz, O. Hirschberg and M. Vucelja, Phys. Rev. X 9, 021060 (2019), doi:10.1103/physrevx.9.021060.
- A. Kumar and J. Bechhoefer, Nature 584, 64 (2020), doi:10.1038/s41586-020-2560-x.
- T. V. Vu and H. Hayakawa, "Thermomajorization Mpemba effect", Phys. Rev. Lett. 134, 107101 (2025), doi:10.1103/physrevlett.134.107101.
- A. Biswas, R. Rajesh and A. K. Pal, "Mpemba effect in a Langevin system: Population statistics, metastability, and other exact results", J. Chem. Phys. 159 (4) (2023) [volume and issue verified; article number to be added by the author], doi:10.1063/5.0155855.
- A. Biswas, V. V. Prasad and R. Rajesh, "Mpemba effect in driven granular gases: role of distance measures", arXiv:2303.10900 (2023).
- H. Hayakawa and S. Takada, "Mpemba effect in a two-dimensional bistable potential", arXiv:2603.24148 (2026).
- G. Teza et al., "Speedups in nonequilibrium thermal relaxation: Mpemba and related effects", Phys. Rep. (2025), doi:10.1016/j.physrep.2025.10.009.
- D. L. Scharfetter and H. K. Gummel, IEEE Trans. Electron Devices 16, 64 (1969), doi:10.1109/t-ed.1969.16566 [DOI verified; authors and volume from memory].
- W. K. Grassmann, M. I. Taksar and D. P. Heyman, Oper. Res. 33, 1107 (1985), doi:10.1287/opre.33.5.1107 [title and DOI verified; authors from memory].
- [Other references of the earlier draft are to be re-verified by the author before use; item 9 there has an erratum, Phys. Rev. Lett. 128, 229901 (2022); items 4 and 11 could not be verified.]

## Appendix: timeline of the pre-specification (local git, one machine, not pushed)

`3e002b6` baseline; `db3de4a` tag `prereg-v2-frozen` (criteria, P0, K0–K4); `2ac799e` solver-method deviation and validation tolerances (before the solver code); `ce807a5` solver and validation gates;
`7d71290`, `dc145c9` main-run protocol, controls and spectral criterion (before the corresponding code and runs); `bde4119` (sweep) and `6c96eb9` (attribution, W1 control, equivalence check), each before its code.
Tolerances for the simulation cross-check were written into the file before its code but committed together with it. All post-hoc additions are listed in `PREREG_v2.md`, sections D2.7–D2.9.
