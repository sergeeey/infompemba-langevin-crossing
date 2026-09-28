# How a bounded observable and a tail-estimated equilibrium manufacture a Mpemba "crossing": a pre-registered exact-solution check in a double-well Langevin model

**DRAFT — not for submission.** Every bracketed item `[...]` is a decision or fact only the author can supply.

**Author:** [NAME — files of the earlier draft say "Sergei Boiko", the git configuration says "Sergey Boyko"; to be settled by the author]
**Affiliation:** [AFFILIATION]
**Correspondence:** [E-MAIL]
**AI-assistance disclosure:** [text required by the target venue; the analysis code, the numerical validation and the drafting of this note were carried out with an AI coding assistant (Claude); the venue's policy has not been checked]

## Abstract

An earlier draft of this work reported a "Mpemba-like crossing" in the basin-occupancy observable of an overdamped Langevin particle in a
double-well potential, in 94.7% of trajectories at its reference point, and argued that the effect is observable-specific. A pre-publication audit
showed that the analysis could not have detected the effect it claimed. We document why, and re-examine the model with a validated exact solution of
the one-dimensional Fokker–Planck equation and a pre-registered protocol. Three defects are shown to be sufficient to manufacture a crossing:
(i) a bounded observable whose "cold" state already sits at the maximal possible distance from equilibrium, so that no hot state can start farther;
(ii) an equilibrium value estimated from the tails of the two compared trajectories, which is biased by the slower one (mean deviation 0.095 from the
analytic 0.5); and (iii) a crossing criterion that never checks that the hot state started farther. With those removed, a symmetric hot state crosses
a one-well-restricted cold state in the Kullback–Leibler and Wasserstein-1 distances at every one of 288 parameter combinations for which the
precondition holds (283 of 288 in Wasserstein-1; the other five miss a preset threshold by 0.001), and the crossing is predicted in all 224 eligible
cases by the known criterion of Lu and Raz, that is, by the overlap of the initial state with the slowest relaxation mode. The mechanism is therefore
not new, and closely related results exist [Hayakawa & Takada 2026; Biswas et al. 2023]; the contribution here is methodological: a checklist of
failure modes, a validated exact solver, and a pre-registered protocol with positive and negative controls.

## 1. Introduction

The Mpemba effect, in the Markovian setting, is the statement that a system prepared farther from equilibrium can relax faster than one prepared
closer to it [Lu & Raz 2017; Klich et al. 2019]. Its spectral origin — the coefficient of the slowest relaxation mode in the initial state — is
established [Lu & Raz 2017; Klich et al. 2019; Vu & Hayakawa 2025], and it has been observed with a colloidal particle in a double-well potential
[Kumar & Bechhoefer 2020]. Dependence of the effect on the choice of distance measure has been shown in granular gases [Biswas et al. 2023], and an
analytically solvable model of overdamped Langevin dynamics in a two-dimensional bistable potential, with Kullback–Leibler crossing conditions, has
appeared very recently [Hayakawa & Takada 2026]. See [Teza et al. 2025] for a review.

We do not claim a new mechanism. This note reports what happened when an analysis of exactly this kind was carried out with a metric that could not
fail, how that was found, and what a corrected, validated analysis of the same model shows. We report the earlier analysis as a case study because the
failure modes are generic.

## 2. Model, exact solver and validation

**Model.** Overdamped Langevin dynamics dx = −U′dt + √(2T)dW with U(x,y) = b(x²−1)² + κ·b·x + 0.3y²; the y coordinate is an independent
Ornstein–Uhlenbeck process, so the problem reduces to the x marginal. Parameters: barrier b ∈ {0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5}, temperature
T ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1}, tilt κ ∈ {0, 0.02, 0.05, 0.1} (288 combinations, b/T from 0.1 to 100).

**Initial states.** Cold: the Boltzmann distribution restricted to x > 0 (all probability in the right well, locally equilibrated). Hot: a Gaussian
N(0, 3²) in x. The hot state was chosen by a rule applied to t = 0 distances only (Section 3), before any dynamics were computed.

**Solver.** A Scharfetter–Gummel finite-volume generator [Scharfetter & Gummel 1969] with implicit Euler steps on a geometric time grid; the tridiagonal
solve is rewritten so that every update adds positive numbers only, which keeps componentwise relative accuracy at any step size and makes the
density non-negative by construction. Two grid steps (dx = 0.01, 0.005) and two time-step ratios give Richardson-extrapolated distances and an error
estimate. A spectral solver was rejected because projecting a wide hot state onto modes requires exp(U/2T), which exceeds double precision in the tails
(exponent about 1.4·10⁴ for b = 5, T = 0.05, |x| = 4.2).

**Validation (tolerances fixed before the code was written; failed early runs are recorded in the repository).** Against the exact Ornstein–Uhlenbeck
solution: relative error ≤ 7.7·10⁻⁴ in KL and W1 at every one of 1163 recorded times (tolerance 5·10⁻³). Discrete stationarity, mass conservation and
positivity at every one of 17 290 steps: 4.3·10⁻¹⁴, 4.0·10⁻¹⁴ and exactly non-negative. The slowest rate λ₁ from first-passage quadrature agrees with the
symmetrised-generator eigenvalue within a factor 1.079 at all 224 grid points with b/T ≤ 12. An independent Euler–Maruyama simulation (10⁵ particles,
three fixed seed bases) reproduces λ₁ within 7.9% (tolerance 10%) and the Wasserstein-1 curves within 0.0105 (tolerance 0.02). The KL evaluation
needed a series form for small relative deviations to remove a 10⁻¹⁷ noise floor. [Some margins are thin: 1.3× for λ₁, 1.05× for grid refinement in KL.]

## 3. Pre-registered protocol

Criteria were committed to version control before the corresponding code and runs (repository tags and commits listed in the Appendix).

**Precondition P0.** In a given metric, the hot state must start at least 1.25 times as far from equilibrium as the cold state. A grid point that fails P0
is labelled NO_TEST, never "no effect".

**EFFECT.** For distance D: the gap D_cold − D_hot must exceed five times the summed discretisation errors at every time from a moment t* on, with
t* < t_max/1.05, where t_max = 8/λ₁ and times with D_cold < 10⁻¹⁸ are excluded. An effect in the headline sense requires this in both KL and W1, so that a
crossing produced only by the tail sensitivity of KL is not counted.

**Controls.** Positive (G3): symmetric hot states at b/T = 10, κ = 0 must be detected (10/10 were). Negative (G4): a pair (cold evolved to s*, cold at
t = 0) cannot cross by the data-processing inequality for the discrete reversible chain, so the pipeline must not report a crossing (8/8 did not). The
originally planned negative control, "the same state but wider inside the same well", was replaced before running, because it is not guaranteed to be
negative: the cold state satisfies ρ/π − 1 = sign(x), and a wider state in the same well puts more weight near the barrier where the left eigenfunction is
smaller, so its overlap with the slow mode can be smaller and a crossing is then legitimate.

**Spectral criterion R.** With g₁ = φ₁/√π the left eigenvector of the slowest mode, c₁(ρ) = Σᵢ g₁ᵢρᵢ; R predicts that the hot state is asymptotically
closer than the cold one if and only if |c₁(hot)| < |c₁(cold)|. It was checked independently against the late-time KL amplitude ½c₁²e^(−2λ₁t) (within 5%).

**Decision rules (D2.5, fixed in advance).** K1: fewer than 10 EFFECT points among the P0-eligible κ = 0 points ⇒ reject. K2: EFFECT nowhere in both
metrics but in KL ⇒ metric artefact. K3: R agrees with the direct calculation in at least 95% of eligible points and K1, K2 pass ⇒ the effect is an
instance of the known criterion and is not presented as a new mechanism. All thresholds are heuristics fixed before the data and are not calibrated.

## 4. Results

**Grid.** KL: EFFECT at 288 of 288 combinations. W1: EFFECT at 283 and NO_TEST at 5 (all at κ = 0.1, P0 ratio 1.249 against the threshold 1.25). K1 passed
(72 of 72 at κ = 0), K2 passed. The crossing occurs early, at t*/t_max between 7·10⁻⁴⁵ and 1.7·10⁻² (KL; median 5·10⁻⁴). The smallest margin over the
discretisation error is 1.68 (gap/(5·error)) at b/T = 100, and there is no region of the (b, T) plane where the effect is absent among the P0-eligible points.

**Mechanism.** At κ = 0 the symmetric hot state has |c₁| ≤ 1.8·10⁻⁹ (zero by symmetry) whereas the cold state has |c₁| between 0.90 and 1.00. With tilt,
r = |c₁(hot)|/|c₁(cold)| lies between 0.003 and 0.43. R agreed with the direct late-time gap in 224 of 224 eligible points. Because r < 1 everywhere in that
set, the agreement could not fail in the opposite direction; a post-hoc check with pairs constructed so that R predicts no crossing (r from 5.6 to 10⁹) agreed
in 8 of 8. The mechanism is the known one, in its simplest, symmetry-forced form.

**Other observables (descriptive).** At κ = 0 the cold state sits exactly at the equilibrium value of ⟨U⟩ (its distance is 3·10⁻¹⁷, round-off), so the
energy comparison is degenerate there: no statement of the form "energy shows no crossing" can be tested. At κ ≠ 0, ⟨U⟩ crosses at 216 of 216 points, and
Var(x) crosses at 288 of 288.

## 5. What went wrong in the earlier analysis (case study)

1. **A bounded observable started at its maximum.** For p_left the equilibrium is 0.5 and the distance is at most 0.5; a cold state fully in one well starts at
   0.5. The hot state cannot start strictly farther, so the P0 precondition is unsatisfiable (0 of 72 points), and a "crossing" of the earlier kind is only a
   statement that the hot state started closer.
2. **The equilibrium was estimated from the compared trajectories.** Taking O_eq as the mean of the last 10% of both trajectories, with the cold trajectory
   not yet relaxed, gave a mean deviation of 0.095 (maximum 0.25) from the analytic 0.5 on exact trajectories, matching the deviation found in the original
   simulations (0.097 and 0.259). With this O_eq the distance gap tends to zero in the tail by construction, and its sign is set by whether the cold trajectory
   is still below its own tail mean.
3. **No check of the initial ordering.** The criterion counted the fraction of time the hot state was closer; with the hot state closer from the start this is
   near 100%. On exact trajectories the first "crossing" was at the first record in 70 of 72 points (646 of 720 in the original runs).
4. **The controls could not fail on the trivial reading.** The four controls (balanced cold state, single-well potential, no gradient clipping, 30 seeds)
   each behaved as the mechanism predicted, but none of them tested whether the pipeline reports a crossing when the hot state simply starts closer — which
   is what the metric was measuring.

**A reproduction that failed.** Applying the earlier metric to noise-free exact trajectories did not reproduce the earlier phase map (Pearson r = 0.33; 57 versus
27 points above 70%). Candidate causes — finite-sample noise, gradient clipping, sensitivity of the metric to noise-level differences — were not tested, and we
do not claim that any of them is the explanation.

**Untested claims in the earlier draft.** A negative result for a neural network was presented as evidence for the mechanism; the experiments were invalid
(the "cold" state was not stationary, its KL drifted by +7400%), so the result is "not tested", and this note makes no statement about neural networks.

## 6. Limitations

One-dimensional reduction with a common y coordinate; one hot and one cold family; a single potential family; Markovian overdamped dynamics; KL and W1 as the
headline metrics. All thresholds are pre-set heuristics; the P0 ratio 1.25 was missed by 0.001 at five points. Margins are thin in the extreme corner (1.68 at
b/T = 100) and the spatial error at dx = 0.01 is about 1% in KL. For b/T > 12 (64 points) λ₁ is a first-passage estimate, validated only where an eigenvalue
computation is possible. Validation against simulation used three points at κ = 0. Code review was performed by an AI reviewer reading the code; no
independent human replication exists.

## 7. Conclusion

In this model the symmetric hot state crosses the one-well cold state wherever the precondition holds, because it does not excite the odd slow mode; this is
the known spectral mechanism. The earlier reported "phase diagram" and its "observable-specificity" were products of an analysis that could not fail. The
practical checklist: check the initial ordering; use an analytic or independently converged equilibrium; check that the observable's distance can exceed the
cold state's initial distance; check for degeneracy when the cold state is at the observable's equilibrium value; include a control the metric must fail.

## Data and code availability

[Repository URL and archive DOI to be supplied by the author. At present the repository is local only; the pre-registration tag `prereg-v2-frozen` and all
commits have not been pushed to any remote, so the timestamps in the Appendix are local.] Per-configuration summary: `prereg_v2/summary_main.csv`; manifest
of the 288 curve files with SHA-256: `prereg_v2/out_manifest.csv`.

## References (verified in OpenAlex/Crossref; details in `prereg_v2/N1_LITERATURE.md`)

- Z. Lu and O. Raz, Proc. Natl. Acad. Sci. USA 114, 5083 (2017), doi:10.1073/pnas.1701264114.
- I. Klich, O. Raz, O. Hirschberg and M. Vucelja, Phys. Rev. X 9, 021060 (2019), doi:10.1103/physrevx.9.021060.
- A. Kumar and J. Bechhoefer, Nature 584, 64 (2020), doi:10.1038/s41586-020-2560-x.
- T. V. Vu and H. Hayakawa, Phys. Rev. Lett. 134, 107101 (2025), doi:10.1103/physrevlett.134.107101.
- G. Teza et al., "Speedups in nonequilibrium thermal relaxation: Mpemba and related effects", Phys. Rep. (2025), doi:10.1016/j.physrep.2025.10.009.
- A. Biswas, V. V. Prasad and R. Rajesh, "Mpemba effect in driven granular gases: role of distance measures", arXiv:2303.10900 (2023).
- H. Hayakawa and S. Takada, "Mpemba effect in a two-dimensional bistable potential", arXiv:2603.24148 (2026).
- D. L. Scharfetter and H. K. Gummel, IEEE Trans. Electron Devices 16, 64 (1969), doi:10.1109/t-ed.1969.16566 [DOI verified; authors and volume from memory].
- [Other references of the earlier draft to be re-verified by the author before use; item 9 there has an erratum, PRL 128, 229901 (2022).]

## Appendix: timeline of the pre-registration (local git, one machine, not pushed)

`3e002b6` baseline; `db3de4a` tag `prereg-v2-frozen` (criteria, P0, K0–K4); `2ac799e` solver-method deviation and validation tolerances (before the solver code);
`ce807a5` solver and validation gates; `7d71290`, `dc145c9` protocol of the main run, controls and the spectral criterion (before the corresponding code and runs).
Tolerances for the simulation cross-check were written into the file before its code but committed together with it. Post-hoc additions after the main run are
listed in `PREREG_v2.md`, section D2.7.
