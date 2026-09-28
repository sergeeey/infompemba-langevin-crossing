# A Mpemba "crossing" that the metric could not fail to report: an audit of an earlier analysis and a pre-specified re-analysis of a double-well Langevin model

**DRAFT v0.3 — not for submission.** Bracketed items `[...]` are decisions or facts only the author can supply.
**Intended venue class:** an archival note (e.g. Zenodo, possibly with a replacement/withdrawal notice for the earlier draft). The frozen pre-specification (rule K3) sets exactly this ceiling: "not submitted as new work; at most a note that presents the result as a numerical illustration". Sending this to a research journal would need a declared deviation from that rule.

**Author:** [NAME — the files of the earlier draft say "Sergei Boiko", the git configuration says "Sergey Boyko"; to be settled by the author]
**Affiliation:** [AFFILIATION] **Correspondence:** [E-MAIL]
**AI-assistance disclosure:** [text required by the target venue. The analysis code, the numerical validation, the pre-specification and the drafting of this note were all produced with one AI coding assistant (Claude) in a single working session on one machine, under the author's direction; reviewers were also AI agents. The venue's policy has not been checked.]
**Status of the earlier draft:** [AUTHOR TO STATE: whether the earlier draft was ever posted or circulated, and where.]

## Abstract

An earlier draft of this work reported a "Mpemba-like crossing" in the basin-occupancy observable p_left of an overdamped Langevin particle in a double-well potential
(94.7% of the recorded time at its reference point, b = 1, T = 0.2) and argued that the effect is observable-specific. A pre-publication audit showed that the analysis could not have detected the effect it
claimed. We document why and re-examine the model with a grid-refined, validated numerical solution of the one-dimensional Fokker–Planck equation and a pre-specified protocol with controls.
(1) *Why a crossing was always reported.* For p_left at κ = 0 (no tilt) the symmetric hot state is at the equilibrium value 0.5 from the start, so it can never start farther than the cold state; the criterion
did not check the initial ordering. On noise-free numerical trajectories this alone gives a "crossing" (> 70% of the time) at 72 of 72 grid points even with the analytic equilibrium; with the initial-ordering
check none of the 72 points can be tested at all. Estimating the equilibrium from the tails of the compared trajectories is not necessary for the artefact. (2) *The earlier phase map.* The earlier metric on
noise-free trajectories does not reproduce the earlier map (Pearson r = 0.33), and is worse than predicting the map's mean everywhere (mean error 24.7 versus 18.5 percentage points). At 57 of 72 points most records have a
true gap below the sampling noise; a post-hoc model that treats those records as coin flips reduces the error to 8.8 points (best 6.9 over a range of thresholds; the noise floor of the target is about 3.5), so it explains the map only
partly. (3) *Re-analysis.* With the flaws removed, a symmetric hot state crosses a one-well-restricted cold state in the Kullback–Leibler distance at all 288 parameter combinations, and in both Kullback–Leibler and
Wasserstein-1 at 283 (at five points the Wasserstein-1 precondition is missed by a heuristic threshold, by about 0.001). This result is largely determined by the design: at κ = 0 the hot state has no overlap with the odd slowest
mode, and at κ ≠ 0 (up to 0.1) the overlap ratio to the cold state stays between 0.003 and 0.43, so the agreement with the known spectral criterion of Lu and Raz (224 of 224 evaluated points) is a consistency check that
could not have failed in the other direction. The result depends strongly on the hot state's width: in a sweep at b/T = 10 the Wasserstein-1 effect holds in 70 of 70 testable runs with a wide hot state and 0 of 20 testable runs with a narrow one.
The mechanism is not new and closely related results exist; the contribution here is methodological.

## 1. Introduction

The Markovian Mpemba effect is the statement that a system prepared farther from equilibrium can relax faster than one prepared closer to it [Lu & Raz 2017; Klich et al. 2019]; its spectral origin — the
coefficient of the slowest relaxation mode in the initial state — is established there. In those works the compared states are thermal states at different temperatures; here the states are engineered (a Gaussian and
a one-well-restricted Boltzmann distribution), and the thermal candidates we tried failed a precondition in W1, so we call the phenomenon a "Mpemba-like anomalous ordering" and use the spectral criterion in its general
form (ordering of slow-mode coefficients). A formulation independent of the distance measure exists [Vu & Hayakawa 2025]. Closely related to the present model: the metastable Mpemba effect [Chétrite, Kumar &
Bechhoefer 2021] and its observation for a colloidal particle in a double-well potential [Kumar & Bechhoefer 2020]; an exact spectral treatment of a piecewise-linear double well concluding that neither metastability nor
asymmetry is necessary or sufficient and that initial population statistics and the observable matter [Biswas, Rajesh & Pal 2023]; an exact single-well example without metastable minima [Biswas & Rajesh 2023]; the
dependence on the distance measure in granular gases [Biswas, Prasad & Rajesh 2023]; and an analytically solvable two-dimensional bistable model with Kullback–Leibler crossing conditions [Hayakawa & Takada 2026].
See [Teza et al. 2025] for a review. (Only the abstracts of the 2021–2026 works were read.)

We do not claim a new mechanism. We report what happened when an analysis of this kind was carried out with a metric that could not fail, how that was found, and what a corrected, validated analysis shows. The earlier
analysis is one case; we make no claim about how common its defects are.

## 2. Model, numerical solution and validation

**Model.** Overdamped Langevin dynamics dx = −U′dt + √(2T)dW with U(x, y) = b(x²−1)² + κ·b·x + 0.3y². The y coordinate is an independent Ornstein–Uhlenbeck process, so all distances used here are distances of the
x marginal, and the initial y distribution does not enter. (The earlier draft's cold and hot states also had different y distributions; results below concern the x-marginal reduction, not an exact repetition of it.)
Parameters: b ∈ {0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3, 5}, T ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1}, κ ∈ {0, 0.02, 0.05, 0.1} (288 combinations, b/T from 0.1 to 100). The x grid extends to |x| = 15 with reflecting walls
(the mass of N(0, 3²) beyond |x| = 15 is 6·10⁻⁷; the grid state is the normalised discretised Gaussian).

**Initial states.** Cold: the Boltzmann distribution restricted to x > 0. Hot: a Gaussian N(0, 3²) in x in the main run, fixed by rule H (Section 3) using distances at t = 0 only, before dynamics were computed.

**Numerical solution.** A Scharfetter–Gummel finite-volume generator [Scharfetter & Gummel 1969] with implicit Euler steps on a geometric time grid. The tridiagonal solve is rewritten so that every update adds positive
numbers only, in the spirit of subtraction-free (GTH-type) elimination [Grassmann, Taksar & Heyman 1985]; this makes the density non-negative by construction and, in our tests, preserved the slow modes at step sizes where
ordinary banded elimination did not. We have no error bound. Two grid steps (dx = 0.01, 0.005) and two time-step ratios give Richardson-extrapolated distances and an error estimate; this is an estimate, not a proof, and the
spatial error at dx = 0.01 is about 1% in KL. At dx = 0.005 every well has at least 7 cells per width √(T/8b). A spectral solver was rejected because projecting a wide hot state onto modes needs exp(U/2T), which
exceeds double precision in the tails (exponent about 1.4·10⁴ for b = 5, T = 0.05, |x| = 4.2).

**Validation (tolerances written before the code; failed early runs and their fixes are recorded in `PREREG_v2.md` D1.1–D1.4).** Against the analytic Ornstein–Uhlenbeck solution: relative error ≤ 7.7·10⁻⁴ in KL and W1
at all 1163 recorded times (tolerance 5·10⁻³). Stationarity, mass conservation and positivity at all 17 290 steps of a test run: residuals 4.3·10⁻¹⁴ and 4.0·10⁻¹⁴, density non-negative. The slowest rate λ₁ from first-passage
quadrature agrees with the symmetrised-generator eigenvalue within a factor 1.079 at all 224 grid points with b/T ≤ 12. An Euler–Maruyama simulation (10⁵ particles, three fixed seed bases, three points at κ = 0) reproduces
λ₁ within 7.9% (tolerance 10%) and the Wasserstein-1 curves within 0.0105 (tolerance 0.02). Three gates failed on first runs, for four reasons (first-order time error; ill-conditioning of banded elimination at large steps;
a cancellation floor of 10⁻¹⁷ in the KL evaluation; Euler error in the hot tails of the cross-check simulation). Each was fixed in the *method* with the tolerance unchanged; in the last case the object changed is the
reference simulation (its step-splitting limit dt·U″ from 0.5 to 0.05), justified by a second route (dt reduced fivefold). Two margins are thin: 1.3× for the simulation cross-check of λ₁ and 1.05× for the grid refinement in KL.

## 3. Pre-specified protocol

Criteria were committed to local version control before the corresponding code and runs, except where the Appendix says otherwise. All commits are on one machine, dated the same day, and have not been pushed to an
external timestamping service; the ordering therefore cannot at present be verified by a third party.

**P0 (a heuristic filter, not a theorem), applied per metric.** In a given metric the hot state must start at least 1.25 times as far from equilibrium as the cold state; a point failing it is labelled NO_TEST, never
"no effect". The frozen text defined P0 jointly (KL and W1); the labelling below applies it per metric (KL at 288 points, W1 at 283), while the decision rule K1 uses points where it holds in both. The value 1.25 is uncalibrated.

**EFFECT.** With gap = D_cold − D_hot: gap > 5 × (summed discretisation errors of both states) at every recorded time from a moment t* on, with t* < t_max/1.05, t_max = 8/λ₁, excluding times where D_cold < 10⁻¹⁸. (t*
is the first record after the last time at which the condition fails, not the crossing time.) EFFECT_BOTH means EFFECT in both KL and W1. CROSSING_NO_MARGIN means gap > 0 somewhere but the condition above fails.

**Rule H (hot state).** Candidates in order: Gaussian σ = 2 (the earlier draft's), Gaussian σ = 3, thermal states at 4T and 8T. The first with P0 in both metrics at ≥ 36 of the 72 points at κ = 0 is the main hot state.
Pre-freeze check at t = 0 (no dynamics): σ = 2 → 0/72 (it is closer than the cold state in W1 at every point), σ = 3 → 72/72, thermal → 0/72. Because the outcome at κ = 0 could be predicted from symmetry, the rule protects
against selection less than it may appear.

**Controls.** G3 (positive; 10/10 detected: five Gaussian σ = 3 cases at b/T = 10, κ = 0, EFFECT in both metrics, and five thermal-8T cases, EFFECT in KL, W1 NO_TEST): a subset of the main design, so it tests detection by the
pipeline, not the physics. G4 (negative) for KL: the pair (cold evolved to s*, cold at t = 0), both stepped with the same time-step sequence, cannot cross by the data-processing inequality, because the implicit-Euler step
matrices are functions of the same generator and commute, so A_t = M(s*)B_t with M a stochastic map preserving π (8/8 pairs; maximum KL gap −3·10⁻⁹…−3·10⁻⁸; applied to raw runs, since the guarantee does not cover
Richardson-extrapolated values; P0 in KL holds by construction). It mostly tests bookkeeping and round-off and does not exercise the wide hot state, the heavy tails or the g₁ continuation. G4 for W1: on a convex potential
U = x²/2 the continuous flow is a W1 contraction (3/3 with label NO_CROSSING; maximum raw W1 gap −7.6·10⁻⁵…−7.9·10⁻⁵); this is a continuous-time argument on a different potential, and the pass criterion also allowed
CROSSING_NO_MARGIN. In the double-well G4 pairs the W1 gaps were also negative (−1.4·10⁻⁵…−3.4·10⁻⁵) but no theorem covers W1 there. The originally planned negative control ("the same state but wider inside the same well")
was replaced before running because it is not guaranteed negative: at κ = 0 the cold state has ρ/π − 1 = sign(x), and a wider state in the same well puts more weight near the barrier where the left eigenfunction is smaller,
so its slow-mode overlap can be smaller and a crossing is then legitimate. G5b (post-hoc, 8/8): eight double-well pairs (b/T = 10, κ ∈ {0, 0.05, 0.1}) with a symmetric thermal state as the closer state and the one-well
state as the farther state, where R predicts no crossing (r from 5.6 to 10⁹); the pipeline labelled all eight NO_CROSSING.

**Spectral criterion R.** Let g₁ = φ₁/√π be the left eigenvector of the slowest mode (φ₁ the eigenvector of the symmetrised generator) and c₁(ρ) = Σᵢ g₁ᵢρᵢ, r = |c₁(hot)|/|c₁(cold)|. R predicts that the hot state is
asymptotically closer iff r < 1. Where π < 10⁻²⁰ the ratio φ₁/√π is round-off dominated and g₁ is continued by a constant from the nearest node with π ≥ 10⁻²⁰ (g₁ is flat in the drift-dominated tails); for the wide hot state
this region carries more than 30% of the hot mass in a test case. The late-time KL amplitude ½c₁²e^(−2λ₁t) computed with this c₁ agrees with the time-stepped solution within 5% (κ = 0.05, one case; a consistency check,
since both sides use the same discrete generator). Since r ≤ 0.43 at every evaluated point, no plausible continuation error could flip the sign there, and the 224/224 agreement is not evidence for the continuation.
R was evaluated only where λ₁ is resolvable by an eigenvalue computation and the region π ≥ 10⁻²⁰ is contiguous, i.e. b/T ≤ 12 ("R-eligible": 224 of 288 points, 56 per κ; 168 at κ ≠ 0).

**Decision rules (fixed in advance).** K0: any validation or control gate fails ⇒ stop, no verdict. K1: fewer than 10 EFFECT_BOTH among the κ = 0 points with P0 in both metrics ⇒ reject (not applied if fewer than
36 of 72 satisfy P0). K2: EFFECT in KL but nowhere in both metrics ⇒ metric artefact. K3: R agrees with the direct late-time sign (the sign of the KL gap at the last valid time ≤ t_max) in ≥ 95% of R-eligible points
(excluding r ∈ [0.95, 1.05]) and K1, K2 pass ⇒ the effect is an instance of the known criterion and is not presented as new work (maximum: an archival note). K3′: ≥ 3 disagreements after all gates ⇒ bug search, then
independent review. (K3 and K3′ overlap for 3–11 disagreements; this did not matter here, with 0.) G5: the same agreement restricted to κ ≠ 0, below 95% ⇒ K0. K4: no statement about neural networks may follow from this
model; the frozen text said the earlier machine-learning pilots "were negative"; we now regard them as invalid experiments (the earlier draft's own supplement records that its "cold" state was not stationary, its KL drifting by
+7400%), so the result is "not tested". N1: literature check before any wording of novelty. All thresholds are preset heuristics; the list also includes 10⁻²⁰ (continuation), 3 of 4 (attribution rule), 70% (ablation) and
θ = 0.02 (noise model).

**Post-hoc additions (labelled where they occur).** G5b; a sweep over hot states (definition committed before its code); attribution and degeneracy analyses; the noise-floor model, formulated after the degeneracy prediction P2 failed;
and a correction of a margin statistic (Section 4). Their definitions are in `PREREG_v2.md` D2.7–D2.10 where stated there.

## 4. Results

**Main run.** KL: EFFECT at 288 of 288 points. W1: EFFECT at 283 and NO_TEST at 5 (all at κ = 0.1, P0 ratios of 1.249 against the threshold 1.25 — a miss of about 0.001). K1 passed (72 of 72 at κ = 0), K2 passed (the five
"KL-only" points are those NO_TEST points), G3, G4 and G5 (168 of 168 at κ ≠ 0) passed. *Margins:* by the definition of EFFECT, the ratio gap/(5·error) at t* is just above 1, so the minimum over a window that
includes t* is about 1 by construction; the informative statistics are the median ratio over the sustained window (at least 165 in KL and 377 in W1 at every point), the fraction of window records with ratio ≥ 2 (at least
0.995) and the ratio at the last valid time (at least 1.7 in KL and 1.9 in W1, in the deepest wells, b = 5, T = 0.05, where λ₁ is only a first-passage estimate and t_max is about 10⁴³). An earlier version of this note and of
the results file reported "smallest margin 1.68"; that figure is the late-time ratio at one point, and the description was wrong. t*/t_max ranges from 7·10⁻⁴⁵ to 1.7·10⁻² (KL): the crossing occurs on the fast intra-well
timescale against a metastable population imbalance.

**The κ = 0 result is guaranteed by symmetry.** At the 56 κ = 0 points with b/T ≤ 12 the symmetric hot state has |c₁| ≤ 1.8·10⁻⁹ while the cold state has |c₁| between 0.90 and 1.00. The residual 10⁻⁹ is not physics: the
computed φ₁ departs from exact oddness by 7·10⁻¹⁰–3.6·10⁻⁹ and |c₁(hot)| tracks half of that, three orders below the a-priori eigenvector error bound ε‖S‖/λ₁ ≈ 4–8·10⁻⁶. The 72 κ = 0 points therefore record one fact
72 times: they check the implementation, and finding the effect at every one of them is not evidence about the model. (The EFFECT label additionally needs the 5× margin within t_max; the asymptotic ordering is what is
guaranteed.) With tilt the hot state is no longer orthogonal to the slow mode, but r stays small (0.003–0.43) because the cold state, restricted to one well, has |c₁| near its maximum, so almost any hot state that spreads
into the other well has r < 1. The absence of any (b, T) region without the effect among the P0-satisfying points therefore follows from the design (the computed r ≤ 0.43 and the choice of cold state), not from
continuity alone, and it includes shallow barriers (b = 0.1, T = 1) where there is no metastability, consistent with [Biswas, Rajesh & Pal 2023; Biswas & Rajesh 2023].

**Consistency of R with the direct calculation.** R and the direct late-time sign agreed at 224 of 224 R-eligible points (56 at κ = 0, where agreement is guaranteed by symmetry; 168 at κ ≠ 0). This is a consistency check
and not confirmation of the criterion: both come from the same discrete generator, the asymptotic ordering is fixed by the slowest-mode coefficient, and r < 1 at every eligible point, so the r > 1 → no-crossing direction
was not tested by this set. The post-hoc pairs of G5b (r from 5.6 to 10⁹) agreed in 8 of 8.

**Sensitivity to the hot state (post-hoc sweep at b/T = 10 only; 225 runs = 5 contexts × κ ∈ {0, 0.05, 0.1} × 15 Gaussian hot states, μ ∈ {−2, −1, 0, 1, 2}, σ ∈ {1, 1.5, 3}; cold state unchanged).** P0 in KL holds in
all 225 runs (KL P0 is not a real filter here; the W1 P0 is). EFFECT in KL: 55/75 (σ = 1), 65/75 (σ = 1.5), 75/75 (σ = 3); the remaining runs are CROSSING_NO_MARGIN (30 in total: hot states placed in one well with small
width, where r approaches 1 — these are among the 20 runs with |r − 1| ≤ 0.05 together with others [not resolved run by run]). W1: 55, 55 and 5 of the runs are NO_TEST; among the testable runs W1 EFFECT holds in 0 of 20
(σ = 1), 10 of 20 (σ = 1.5) and 70 of 70 (σ = 3). So the "both metrics" headline is a property of a wide hot state. R and the direct sign agreed in 225 of 225 runs without excluding r ∈ [0.95, 1.05]; r < 1 in all 225,
so this sweep also does not probe r > 1.

**Other observables (descriptive).** At κ = 0 the cold state sits exactly at the equilibrium value of ⟨U⟩ (its distance is 3·10⁻¹⁷, round-off), so the energy comparison is degenerate there: "energy shows no crossing" cannot
be tested. At κ ≠ 0, ⟨U⟩ crosses at 216 of 216 points and Var(x) at 288 of 288.

## 5. Case study: the earlier analysis

**The earlier metric.** For each observable O the earlier analysis took O_eq as the mean of the last 10% of the two compared trajectories, computed gap_v1 = D_hot − D_cold along the run (note: the sign convention is
opposite to Section 3) and reported the fraction of recorded time with gap_v1 < 0 ("% crossed"). It did not check that the hot state started farther. For p_left the equilibrium is 0.5 at κ = 0 and the earlier grid had κ = 0
only; a cold state fully in one well starts at the maximal distance 0.5, so P0 is unsatisfiable (0 of 72 points). (With tilt this argument no longer holds.)

**Ablation on noise-free numerical trajectories (72 points, κ = 0, the earlier initial states, x marginal):**

| Variant | Points with "crossing" > 70% of the recorded time | Mean |
|---|---|---|
| earlier metric (tail-estimated O_eq, no P0 check) | 57 / 72 | 82.4% |
| analytic O_eq = 0.5, still no P0 check | 72 / 72 | 99.8% (minimum 88.0%) |
| either estimate + P0 check | none of the 72 points can be tested (P0 unsatisfiable) | — |

Row 2 is a statement about the design, not a discovery: with the analytic equilibrium the symmetric hot state has D_hot ≡ 0 for p_left, so it is "closer" from the first record (at the first record in 70 of 72 points) without any
crossing having happened. The tail-estimated equilibrium is not necessary and lowers the fraction. Its own bias is real: mean deviation 0.095 (maximum 0.25) from 0.5, consistent with 0.097 and 0.259 in the original
simulations. At the earlier reference point (b = 1, T = 0.2, a non-degenerate point) the noise-free metric gives 95.0% against 94.8 ± 0.6% in the earlier ten-seed runs, so the headline figure is reproduced. This ablation
explains why a crossing was reported; it does not explain the shape of the earlier map, nor the earlier "observable-specificity" claim.

**The earlier phase map is not reproduced by its own metric on noise-free dynamics** (Pearson r = 0.33; 57 versus 27 points above 70%; mean absolute discrepancy 24.7 percentage points, worse than the trivial predictor "the map's
mean everywhere", 18.5). *Attribution test (i), defined before it was coded:* rerun the earlier process (Euler–Maruyama step η = 0.005, 30 000 steps, N = 5000 particles, with the gradient clipped to ±100 and the position to ±50
as in the earlier code) at four points chosen by a preset rule (the two largest and the two smallest discrepancies), with N = 5000 and 50 000 and with and without clipping. Finite sample was supported at 1 of 4 points and
clipping at 0 of 4 (threshold 3 of 4), so neither is supported; the test has low power, since two of the four points have almost no discrepancy to close. *Degeneracy hypothesis (ii), formulated after (i) returned "unexplained";
its predictions were written into the analysis script before the script ran but were not part of the committed pre-specification:* with O_eq from the two tails, the gap is zero when both trajectories are stationary
throughout the run (cold stuck near 0, hot at 0.5), and the metric then returns the sign of noise. At the 11 points where the noise-free max|gap| < 10⁻⁶, the earlier ten-seed standard deviation is 51.4 percentage points against
4.7 elsewhere, and 100% of the individual earlier runs there lie at an extreme (< 10% or > 90%) — as predicted; the prediction that reproduction would be good away from those points failed (mean discrepancy 19.8 there).
*Post-hoc noise-floor model (iii), formulated after (ii) partly failed:* for each run the sign of gap_v1 in records with |gap_v1| ≤ θ is effectively determined by sampling noise; treating such records as 50% coin flips (in
expectation; in single runs the sign is largely fixed through the tail-estimated O_eq) and the others as deterministic, with θ = 0.02 taken from the sampling noise of the difference of two fractions of 5000 particles
(≈ 2√2·√(0.25/5000)) and not fitted, gives a mean discrepancy with the earlier map of 8.8 points (r = 0.71); over θ ∈ [0.005, 0.03] it is 6.9–9.8 (r up to 0.82), against baselines of 18.5 (the map's mean everywhere) and 19.1
(50% at the 11 degenerate points and the noise-free metric elsewhere), and a noise floor of the target of about 3.5 (0.8 × the mean standard error, 4.4, of the ten-seed means). At 57 of 72 points more than half of the records have
|gap_v1| ≤ 0.02. The model therefore explains a large part but not all of the map (14 of 72 points remain more than 10 points off).

**Other unsupported claims of the earlier draft.** (a) "No crossing in energy or position variance": at κ = 0 the energy comparison is degenerate (Section 4), and the earlier draft's ⟨x²⟩ and the Var(x) used here are
different observables; the earlier claim was refuted by degeneracy, not by the missing initial-ordering check. (b) A negative result for neural networks was presented as consistent with the mechanism; those experiments were
invalid, so the result is "not tested"; this note makes no statement about neural networks. (c) The controls of the earlier draft (balanced cold state, single-well potential, no gradient clipping, 30 seeds) each behaved as
the mechanism predicted, but none tested whether the pipeline reports a crossing when the hot state simply starts closer.

## 6. Limitations

One-dimensional (x-marginal) reduction; two families of hot and cold states; a single potential family; Markovian overdamped dynamics; KL and W1 as headline metrics; the hot-state sweep at b/T = 10 only and never with r > 1
except in G5b. Thresholds are preset heuristics (Section 3); P0 was missed by 0.001 at five points. The spatial error at dx = 0.01 is about 1% in KL; two grid levels cannot confirm the convergence order. A post-hoc check at dx = 0.0025 at the five points with the smallest late-time ratio (all at b/T ≥ 50, rule fixed
before the run) kept EFFECT in KL and W1 at all five, changed the gap D_cold − D_hot by at most 1.0%, and gave differences between the two finest grids 7–11 times smaller than the two-level error estimate, so that estimate is conservative. For b/T > 12 (64 points) λ₁ is a first-passage estimate validated only where an eigenvalue computation is possible, and R was not
evaluated. The cross-check against simulation used three points at κ = 0. The agreement of R with the direct calculation is a consistency check. The pre-specification is local, from one session, and its author is the
assistant that also ran it. The code was reviewed by AI reviewers reading the code (final verdicts on the solver and on the analysis code: LGTM after fixes; the reviewers could not run their own probes); there is no independent
human replication.

## 7. Conclusion

In this model a symmetric hot state crosses a one-well cold state wherever the precondition holds, because it does not excite the odd slowest mode; this is a known spectral mechanism, the result is largely
determined by the design, and it is not new. The earlier reported crossing was produced by a metric with no check that the hot state starts farther, which by itself yields a "crossing" for p_left at κ = 0; the earlier phase map is
largely explained by the sign of sub-noise differences (post hoc, partly); and the earlier energy claim by degeneracy. Checks that follow from this one case: verify the initial ordering (P0); use an analytic or independently converged
equilibrium; ask whether the observable's distance can exceed the cold state's initial distance; test for degeneracy when the cold state already sits at the observable's equilibrium value; and include a control the metric must fail.

## Data and code availability

[Repository URL and archive DOI to be supplied by the author. At present the repository is local only: the tag `prereg-v2-frozen` and all commits have not been pushed to any remote.] Per-configuration summary:
`prereg_v2/summary_main.csv`; manifest with SHA-256 of the 288 curve files: `prereg_v2/out_manifest.csv`; sweep: `prereg_v2/summary_robustness.csv`.

## References (identifiers verified in OpenAlex/Crossref; see `prereg_v2/N1_LITERATURE.md`)

- Z. Lu and O. Raz, Proc. Natl. Acad. Sci. USA 114, 5083 (2017), doi:10.1073/pnas.1701264114.
- I. Klich, O. Raz, O. Hirschberg and M. Vucelja, Phys. Rev. X 9, 021060 (2019), doi:10.1103/physrevx.9.021060.
- A. Kumar and J. Bechhoefer, Nature 584, 64 (2020), doi:10.1038/s41586-020-2560-x.
- R. Chétrite, A. Kumar and J. Bechhoefer, "The metastable Mpemba effect corresponds to a non-monotonic temperature dependence of extractable work", Front. Phys. (2021), doi:10.3389/fphy.2021.654271.
- T. V. Vu and H. Hayakawa, "Thermomajorization Mpemba effect", Phys. Rev. Lett. 134, 107101 (2025), doi:10.1103/physrevlett.134.107101.
- A. Biswas, R. Rajesh and A. K. Pal, "Mpemba effect in a Langevin system: Population statistics, metastability, and other exact results", J. Chem. Phys. 159 (4) (2023) [article number to be added], doi:10.1063/5.0155855.
- A. Biswas and R. Rajesh, "Mpemba effect for a Brownian particle trapped in a single well potential", Phys. Rev. E 108, 024131 (2023), doi:10.1103/physreve.108.024131.
- A. Biswas, V. V. Prasad and R. Rajesh, "Mpemba effect in driven granular gases: role of distance measures", arXiv:2303.10900 (2023).
- H. Hayakawa and S. Takada, "Mpemba effect in a two-dimensional bistable potential", arXiv:2603.24148 (2026).
- G. Teza et al., "Speedups in nonequilibrium thermal relaxation: Mpemba and related effects", Phys. Rep. (2025), doi:10.1016/j.physrep.2025.10.009.
- D. L. Scharfetter and H. K. Gummel, IEEE Trans. Electron Devices 16, 64 (1969), doi:10.1109/t-ed.1969.16566 [DOI verified; authors and volume from memory].
- W. K. Grassmann, M. I. Taksar and D. P. Heyman, Oper. Res. 33, 1107 (1985), doi:10.1287/opre.33.5.1107 [title and DOI verified; authors from memory].
- [Other references of the earlier draft are to be re-verified by the author before use; item 9 there has an erratum, Phys. Rev. Lett. 128, 229901 (2022); items 4 and 11 could not be verified.]

## Appendix: timeline of the pre-specification (local git, one machine, one session, not pushed)

`3e002b6` baseline; `db3de4a` tag `prereg-v2-frozen` (criteria, P0, K0–K4); `2ac799e` solver-method deviation and validation tolerances (before the solver code); `ce807a5` solver and validation gates;
`7d71290`, `dc145c9` main-run protocol, controls and spectral criterion (before the corresponding code and runs); `bde4119` (sweep), `6c96eb9` (attribution, W1 control, equivalence check), `52b6d7b` (margin correction and
refinement check), each before its code. Tolerances for the simulation cross-check were written into the file before its code but committed together with it, and the degeneracy predictions were written into the analysis script
before it ran but not committed to the pre-specification. All post-hoc additions are listed in `PREREG_v2.md`, sections D2.7–D2.10.
