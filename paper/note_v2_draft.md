# A Mpemba "crossing" that the metric could not test: an audit of an earlier analysis and a pre-specified re-analysis of a double-well Langevin model

**DRAFT v0.4 — not for submission.** Bracketed items `[...]` are decisions or facts only the author can supply.
**Intended venue class:** an archival correction note (e.g. Zenodo, possibly with a replacement/withdrawal notice for the earlier draft). The frozen pre-specification (rule K3, written in Russian; the quotation below is the author's own translation) sets exactly this ceiling: "not submitted as new work; at most a note that presents the result as a numerical illustration". Sending this to a research journal would need a declared deviation from that rule.

**Author:** Sergey Boyko (settled 2026-09-29; the earlier draft's files said "Sergei Boiko", the git configuration said "Sergey Boyko" — the author chose the git-config spelling)
**Affiliation:** Ronin Institute for Independent Scholarship **Correspondence:** sergey.boyko@ronininstitute.org **ORCID:** 0009-0009-2178-5701
**AI-assistance disclosure:** [text required by the target venue. The analysis code, the numerical validation, the pre-specification and the drafting of this note were all produced with one AI coding assistant (Claude) on one machine, under the author's direction, across several working sessions between 2026-09-28 and 2026-09-29 (the later sessions made metadata and cross-check additions and this fourth-pass revision; they did not rerun the underlying numerical pipeline); reviewers were also AI agents. The venue's policy has not been checked.]
**Status of the earlier draft and its data:** [AUTHOR TO STATE whether the earlier draft was ever posted or circulated. Section 5 quotes figures from it and from its result files (`results/phase_diagram.parquet`); a reader can check them only if both are deposited with this note.]

## Abstract

An earlier draft of this work reported a "Mpemba-like crossing" in the basin-occupancy observable p_left of an overdamped Langevin particle in a double-well potential (94.7% of the recorded time at its
reference point b = 1, T = 0.2, over 30 seeds; the ten-seed grid gives 94.8 ± 0.6%) and argued that the effect is observable-specific. A pre-publication audit showed that the analysis could not have detected the
effect it claimed. We document why and re-examine the model with a grid-refined, validated numerical solution of the one-dimensional Fokker–Planck equation and a pre-specified protocol with controls.
(1) *Why a crossing was reported without any possibility of a valid test.* For p_left at κ = 0 (no tilt) the symmetric hot state sits at the equilibrium value 0.5 from the start, so it can never start farther than the cold
state. The earlier draft's own detection definition (§II.D) does require the initial ordering to be reversed, but its analysis code never applies that check — it computes only the fraction of time the gap is negative, with no
initial-state test (§5). (The same draft's §II.B, describing the hot state as "closer to equilibrium basin proportions", is itself in tension with the ordering its own detection formula requires.) On noise-free numerical
trajectories this gap between definition and code alone gives a "crossing" (> 70% of the recorded time) at 70–72 of 72 grid points (mean 98–100%, depending on the
numerical time step) even with the analytic equilibrium; with the initial-ordering check none of the 72 points can be tested. Estimating the equilibrium from the tails of the compared trajectories is not necessary for the artefact.
(2) *The earlier phase map is not reproduced and is not validly explained.* The earlier metric on noise-free trajectories does not reproduce it (Pearson correlation 0.22–0.33 across four time steps; mean discrepancy
24.7–26.2 percentage points, against 18.5 for predicting the map's mean everywhere). At 11 points the metric is degenerate (both trajectories stationary: the earlier ten-seed spread has median 51 points against a median 4.7 elsewhere) and at 57
of 72 points most records have a noise-free difference below the sampling noise; a post-hoc model built on this fits at the original sample size with little advantage over a simple shrinkage baseline (8.8 vs 9.9 percentage points overall), and fails an out-of-sample
check at larger N, so we regard the map as only qualitatively understood. (3) *Re-analysis.* With the flaws removed, a symmetric hot state crosses a one-well-restricted cold state in the Kullback–Leibler distance at all 288 parameter
combinations, and in both Kullback–Leibler and Wasserstein-1 at 283 (at five points the Wasserstein-1 precondition is missed by a heuristic threshold, by about 0.001). This result is largely determined by the design: at κ = 0
the hot state has no overlap with the odd slowest mode (what Klich et al. call the strong Mpemba effect), and at κ ≠ 0 (up to 0.1) the overlap ratio to the cold state stays between 0.003 and 0.43 at the 168 points where it was
evaluated. The agreement with the known spectral criterion (224 of 224 evaluated points) is therefore a consistency check that could not have failed in the other direction. In a sweep at b/T = 10 the Wasserstein-1 effect
depends strongly on the hot state: at |μ| = 2, the only offset at which narrow states are testable in W1, it holds in 0 of 20 runs for σ = 1, 10 of 20 for σ = 1.5 and 30 of 30 for σ = 3. The mechanism is not new and closely related
results exist; this note is a documented correction and case study with a validated re-analysis.

## 1. Introduction

The Markovian Mpemba effect is the statement that a system prepared farther from equilibrium can relax faster than one prepared closer to it [Lu & Raz 2017; Klich et al. 2019]; its spectral origin — the coefficient of the slowest
relaxation mode in the initial state — is, as we recall it, established in those works (we checked their identifiers, and for Klich et al. the abstract, which also defines a strong effect with exponentially faster relaxation; we did not re-read the
full texts). In those works the compared states are thermal states at different temperatures; here the states are engineered (a Gaussian and a one-well-restricted Boltzmann distribution), and the thermal candidates we tried failed a
precondition in W1, so we speak of a "Mpemba-like anomalous ordering" and use the spectral criterion in its general form (ordering of slow-mode coefficients). A formulation independent of the distance measure exists [Vu & Hayakawa 2025].
Closely related to the present model are: the metastable Mpemba effect [Chétrite, Kumar & Bechhoefer 2021] and the colloidal experiment [Kumar & Bechhoefer 2020] (from memory a particle in a potential landscape; abstract not
available to us); an exact spectral treatment of a piecewise-linear double well concluding that neither metastability nor asymmetry is necessary or sufficient and that initial population statistics and the observable matter
[Biswas, Rajesh & Pal 2023]; an exact single-well example without metastable minima [Biswas & Rajesh 2023]; the dependence on the distance measure in granular gases [Biswas, Prasad & Rajesh 2023]; and an analytically solvable
two-dimensional bistable model with Kullback–Leibler crossing conditions [Hayakawa & Takada 2026]. See [Teza et al. 2025] for a review. *What was read:* the abstracts of Klich et al., Vu & Hayakawa, Biswas–Rajesh–Pal, Biswas & Rajesh,
Biswas–Prasad–Rajesh and Hayakawa & Takada; only title and authors of Chétrite et al.; only identifiers of Lu & Raz, Kumar & Bechhoefer and Teza et al. No full text was read.

We do not claim a new mechanism. We report what happened when an analysis of this kind was carried out with a metric that could not test its claim, how that was found and what a corrected, validated analysis shows. The earlier analysis is
one case; we make no claim about how common its defects are.

## 2. Model, numerical solution and validation

**Model.** Overdamped Langevin dynamics dx = −U′dt + √(2T)dW with U(x, y) = b(x²−1)² + κ·b·x + 0.3y². The y coordinate is an independent Ornstein–Uhlenbeck process, so all distances used here are distances of the x marginal, and the
initial y distribution does not enter. (The earlier draft's cold and hot states also had different y distributions; results below concern the x-marginal reduction, not an exact repetition of it.) Parameters: b ∈ {0.1, 0.3, 0.5, 0.7, 1, 1.5, 2,
3, 5}, T ∈ {0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1}, κ ∈ {0, 0.02, 0.05, 0.1} (288 combinations, b/T from 0.1 to 100). The x grid extends to |x| = 15 with reflecting walls (the mass of N(0, 3²) beyond |x| = 15 is 6·10⁻⁷; the grid state is
the normalised discretised Gaussian).

**Initial states.** Cold: the Boltzmann distribution restricted to x > 0. Hot: a Gaussian N(0, 3²) in x in the main run, fixed by rule H (Section 3) using distances at t = 0 only, before dynamics were computed.

**Numerical solution.** A Scharfetter–Gummel finite-volume generator [Scharfetter & Gummel 1969] with implicit Euler steps on a geometric time grid. The tridiagonal solve is rewritten so that every update adds positive numbers only, in the spirit of
subtraction-free (GTH-type) elimination [Grassmann, Taksar & Heyman 1985]; this makes the density non-negative by construction and, in our tests, preserved the slow modes at step sizes where ordinary banded elimination did not. We have no error
bound. Two grid steps (dx = 0.01, 0.005) and two time-step ratios give Richardson-extrapolated distances and an error estimate; this is an estimate, not a proof. The spatial error at dx = 0.01 is about 1% in KL (from a refinement gate at three
shallow points, b/T ≤ 5); at dx = 0.005 every well has at least 7 cells per width √(T/8b). A spectral solver was rejected because projecting a wide hot state onto modes needs exp(U/2T), which exceeds double precision in the tails (exponent about
1.4·10⁴ for b = 5, T = 0.05, |x| = 4.2).

**Validation (tolerances written before the code; failed early runs and their fixes are recorded in `PREREG_v2.md` D1.1–D1.4).** Against the analytic Ornstein–Uhlenbeck solution: relative error ≤ 7.7·10⁻⁴ in KL and W1 at all 1163 recorded
times where D ≥ 10⁻⁶ (tolerance 5·10⁻³). In a test run of 17 290 steps: stationarity residual 4.3·10⁻¹⁴, mass drift 4.0·10⁻¹⁴, density non-negative. The slowest rate λ₁ from first-passage quadrature agrees with the symmetrised-generator eigenvalue
within a factor 1.079 at all 224 grid points with b/T ≤ 12. An Euler–Maruyama simulation (10⁵ particles, three fixed seed bases, three points at κ = 0) reproduces λ₁ within 7.9% (tolerance 10%) and the Wasserstein-1 curves within 0.0105
(tolerance 0.02). Three gates failed on first runs, for four reasons (first-order time error; ill-conditioning of banded elimination at large steps; a cancellation floor of 10⁻¹⁷ in the KL evaluation; Euler error in the hot tails of the cross-check
simulation). Each was fixed in the *method* with the tolerance unchanged; in the last case the object changed is the reference simulation (its step-splitting limit dt·U″ from 0.5 to 0.05), justified by a second route (dt reduced fourfold, from 0.002
to 0.0005). Two margins are thin: 1.3× for the simulation cross-check of λ₁ and 1.05× for the grid refinement in KL.

## 3. Pre-specified protocol

Criteria were committed to local version control before the corresponding code and runs, except where the Appendix says otherwise. All commits were made on one machine, dated the same day. The repository was pushed to GitHub and made public
on 2026-09-29, after all analyses in this note were complete — GitHub's own commit and push timestamps postdate every run, so pushing does not establish that the pre-specification preceded the code it governs; that ordering rests on the
commit dates themselves (Appendix), which were not independently timestamped by a third party before the corresponding run.

**P0 (a heuristic filter, not a theorem), applied per metric.** In a given metric the hot state must start at least 1.25 times as far from equilibrium as the cold state; a point failing it is labelled NO_TEST, never "no effect". The frozen text defined
P0 jointly (KL and W1); the labelling below applies it per metric (KL at 288 points, W1 at 283), while the decision rule K1 uses points where it holds in both. The value 1.25 is uncalibrated.

**EFFECT.** With gap = D_cold − D_hot: gap > 5 × (summed discretisation errors of both states) at every recorded time from a moment t* on, with t* < t_max/1.05, t_max = 8/λ₁, excluding times where D_cold < 10⁻¹⁸. (t* is the first record after the
last time at which the condition fails, not the crossing time.) EFFECT_BOTH means EFFECT in both KL and W1. CROSSING_NO_MARGIN means gap > 0 somewhere but the condition above fails.

**Rule H (hot state).** Candidates in order: Gaussian σ = 2 (the earlier draft's), Gaussian σ = 3, thermal states at 4T and 8T. The first with P0 in both metrics at ≥ 36 of the 72 points at κ = 0 is the main hot state. Pre-freeze check at t = 0 (no
dynamics): σ = 2 → 0/72 (it is closer than the cold state in W1 at every point), σ = 3 → 72/72, thermal → 0/72. Because the outcome at κ = 0 could be predicted from symmetry, the rule protects against selection less than it may appear.

**Spectral criterion R.** Let g₁ = φ₁/√π be the left eigenvector of the slowest mode (φ₁ the eigenvector of the symmetrised generator) and c₁(ρ) = Σᵢ g₁ᵢρᵢ, r = |c₁(hot)|/|c₁(cold)|. R predicts that the hot state is asymptotically closer iff r < 1.
Where π < 10⁻²⁰ the ratio φ₁/√π is round-off dominated and g₁ is continued by a constant from the nearest node with π ≥ 10⁻²⁰ (g₁ is flat in the drift-dominated tails); for the wide hot state this region carries more than 30% of the hot mass
in a test case. The late-time KL amplitude ½c₁²e^(−2λ₁t) computed with this c₁ agrees with the time-stepped solution within 5% (κ = 0.05, one case; a consistency check, since both sides use the same discrete generator). Since r ≤ 0.43 at every
evaluated point, a continuation error would have to reach at least 0.57|c₁(cold)| (from more than 30% of the hot mass) to flip a sign, which the flat-tail argument makes implausible but we have not bounded; the 224/224 agreement is not evidence for the
continuation. R was evaluated only where λ₁ is resolvable by an eigenvalue computation and the region π ≥ 10⁻²⁰ is contiguous, i.e. b/T ≤ 12 ("R-eligible": 224 of 288 points, 56 per κ; 168 at κ ≠ 0; the 48 points at κ ≠ 0 with b/T > 12 were not evaluated).

**Controls.** G3 (positive; 10/10 detected: five Gaussian σ = 3 cases at b/T = 10, κ = 0, EFFECT in both metrics, and five thermal-8T cases, EFFECT in KL, W1 NO_TEST): a subset of the main design, so it tests detection by the pipeline, not the physics.
G4 (negative) for KL: the pair (cold evolved to s*, cold at t = 0), both stepped with the same time-step sequence, cannot cross by the data-processing inequality, because the implicit-Euler step matrices are functions of the same generator and
commute, so A_t = M(s*)B_t with M a stochastic map preserving π (8/8 pairs; maximum KL gap −3·10⁻⁹…−3·10⁻⁸; applied to raw runs, since the guarantee does not cover Richardson-extrapolated values; P0 in KL holds by construction). It mostly tests
bookkeeping and round-off and does not exercise the wide hot state, the heavy tails or the continuation of the slow-mode eigenvector. G4 for W1: on a convex potential U = x²/2 the continuous flow is a W1 contraction (3/3 with label NO_CROSSING;
maximum raw W1 gap −7.6·10⁻⁵…−7.9·10⁻⁵); this is a continuous-time argument on a different potential, and the pass criterion also allowed CROSSING_NO_MARGIN. In the double-well G4 pairs the W1 gaps were also negative (−1.4·10⁻⁵…−3.4·10⁻⁵) but no
theorem covers W1 there. The originally planned negative control ("the same state but wider inside the same well") was replaced before running because it is not guaranteed negative: at κ = 0 the cold state has ρ/π − 1 = sign(x), and a wider state
in the same well puts more weight near the barrier where the left eigenfunction is smaller, so its slow-mode overlap can be smaller and a crossing is then legitimate. G5b (post-hoc, 8/8): eight double-well pairs (b/T = 10, κ ∈ {0, 0.05, 0.1}) with
a symmetric thermal state as the closer state and the one-well state as the farther state, where R predicts no crossing (r from 5.6 to 10⁹); the pipeline labelled all eight NO_CROSSING.

**Decision rules (fixed in advance).** K0: any validation or control gate fails ⇒ stop, no verdict. K1: fewer than 10 EFFECT_BOTH among the κ = 0 points with P0 in both metrics ⇒ reject (not applied if fewer than 36 of 72 satisfy P0). K2: EFFECT in KL
but nowhere in both metrics ⇒ metric artefact. K3: R agrees with the direct late-time sign (the sign of the KL gap at the last valid time ≤ t_max) in ≥ 95% of R-eligible points (excluding r ∈ [0.95, 1.05]) and K1, K2 pass ⇒ the effect is an instance of the
known criterion and is not presented as new work (maximum: an archival note). K3′: ≥ 3 disagreements after all gates ⇒ bug search, then independent review. (K3 and K3′ overlap for 3–11 disagreements; this did not matter here, with 0.) G5: the same
agreement restricted to κ ≠ 0, below 95% ⇒ K0. K4: no statement about neural networks may follow from this model; the frozen text said the earlier machine-learning pilots "were negative"; we now regard them as invalid experiments (the earlier draft's own
supplement records that its "cold" state was not stationary, its KL drifting by +7400%), so the result is "not tested". N1: literature check before any wording of novelty. All thresholds are preset heuristics; the list also includes 10⁻²⁰ (continuation),
3 of 4 (attribution rule), 70% (ablation) and θ = 0.02 (noise model).

**Post-hoc additions (labelled where they occur).** G5b; a sweep over hot states (definition committed before its code); the attribution, degeneracy and noise-model analyses of Section 5; a check of the time-step dependence of the reproduction; and a
correction of a margin statistic (Section 4). Their definitions are in `PREREG_v2.md` D2.7–D2.10 and in `prereg_v2/RESULTS.md` §8–§10 where stated there.

## 4. Results

**Main run.** KL: EFFECT at 288 of 288 points. W1: EFFECT at 283 and NO_TEST at 5 (all at κ = 0.1, P0 ratios of 1.249 against the threshold 1.25 — a miss of about 0.001). K1 passed (72 of 72 at κ = 0), K2 passed (the five "KL-only" points are those NO_TEST
points), G3, G4 and G5 (168 of 168 at κ ≠ 0) passed. *Margins.* The ratio gap/(5·error) exceeds 1 at t* by definition (minimum 1.001, median 2.26 in KL), so a minimum over a window that includes t* is uninformative. The informative statistics: the median
ratio over the sustained window is at least 164.8 in KL and 377 in W1 at every point (records are geometrically spaced, so this weights early times); at least 99.5% of window records have ratio ≥ 2; and the ratio at the last valid time is at least 1.68 in KL and 1.88
in W1, in the deepest wells (b = 5, T = 0.05), where λ₁ is only a first-passage estimate and t_max is about 10⁴³. An earlier version of this note and of the results file called that 1.68 "the smallest margin over the discretisation error"; the number is the
correct smallest late-time ratio, but it is not a minimum margin and the description was wrong (the window it was computed over had also excluded t*). (A subsequent version of this note itself rounded 1.68 and 1.88 up to "1.7" and "1.9" and stated them as a
lower bound rather than the exact minimum — a fourth-pass review caught this; the numbers here are now exact, not rounded.) A check at dx = 0.0025 at the five points with the smallest late-time ratio (all at b/T ≥ 50; rule fixed before the
run) kept EFFECT in KL and W1 at all five, changed the gap D_cold − D_hot by at most 1.0%, and gave differences between the two finest grids 7–11 times smaller than the main run's error estimate (which has spatial and time-Richardson parts), so that
estimate is conservative. t*/t_max ranges from 7·10⁻⁴⁵ to 1.7·10⁻² (KL): the crossing occurs on the fast intra-well timescale against a metastable population imbalance.

**The κ = 0 result is guaranteed by symmetry.** At the 56 κ = 0 points with b/T ≤ 12 the symmetric hot state has |c₁| ≤ 1.8·10⁻⁹ while the cold state has |c₁| between 0.90 and 1.00. The residual 10⁻⁹ is not physics: the computed φ₁ departs from exact oddness
by 7·10⁻¹⁰–3.6·10⁻⁹ and |c₁(hot)| tracks half of that, three orders below the a-priori eigenvector error bound ε‖S‖/λ₁ ≈ 4–8·10⁻⁶. The 72 κ = 0 points therefore record one fact 72 times: they check the implementation, and finding the effect at every one of them is not
evidence about the model. (The EFFECT label additionally needs the 5× margin within t_max; the asymptotic ordering is what is guaranteed.) The guarantee also rests on the choice of cold state: its slow-mode coefficient is large because it is restricted to one well. A symmetric cold state (for example a thermal state at a lower temperature) would have c₁ = 0 by the same parity argument, and this mechanism would then give no crossing. A separate spectral toy check by a different AI agent (b/T = 4, symmetric thermal cold state at T = 0.35 versus a hot state at T = 2; script not part of this repository, run once here) gave |c₁| ≈ 2.6·10⁻¹³ for both states and no crossing above the numerical floor, as this argument predicts; it does not test the present design. With tilt the hot state is no longer orthogonal to the slow mode, but at the 168 evaluated κ ≠ 0 points r stays small
(0.003–0.43) because the cold state, restricted to one well, has |c₁| near its maximum, so almost any hot state that spreads into the other well has r < 1. For the 48 κ ≠ 0 points with b/T > 12 this design explanation is inferred, not computed. The absence of a (b, T) region
without the effect among the P0-satisfying points therefore follows from the design (the computed r ≤ 0.43 and the choice of cold state), not from continuity alone, and it includes shallow barriers (b = 0.1, T = 1) where there is no metastability, consistent with
[Biswas, Rajesh & Pal 2023; Biswas & Rajesh 2023].

**Consistency of R with the direct calculation.** R and the direct late-time sign agreed at 224 of 224 R-eligible points (56 at κ = 0, where agreement is guaranteed by symmetry; 168 at κ ≠ 0). This is a consistency check and not confirmation of the criterion: both
come from the same discrete generator, the asymptotic ordering is fixed by the slowest-mode coefficient, and r < 1 at every eligible point, so the r > 1 → no-crossing direction was not tested by this set. The post-hoc pairs of G5b (r from 5.6 to 10⁹) agreed in 8 of 8.

**Sensitivity to the hot state (post-hoc sweep at b/T = 10 only; 225 runs = five (b, T) pairs with b/T = 10 — (2, 0.2), (1, 0.1), (3, 0.3), (5, 0.5), (0.5, 0.05) — × κ ∈ {0, 0.05, 0.1} × 15 Gaussian hot states with μ ∈ {−2, −1, 0, 1, 2}, σ ∈ {1, 1.5, 3}; cold
state unchanged).** P0 in KL holds in all 225 runs (KL P0 is not a real filter here; the W1 P0 is). EFFECT in KL: 55/75 (σ = 1), 65/75 (σ = 1.5), 75/75 (σ = 3); the remaining 30 runs are CROSSING_NO_MARGIN, all at |μ| = 2 (20 with σ = 1, 10 with σ = 1.5), with r from 0.87 to
0.97, 20 of them with |r − 1| ≤ 0.05. W1: narrow states are testable only at |μ| = 2 (W1 NO_TEST in 55 of 75 runs for σ = 1 and for σ = 1.5, in 5 of 75 for σ = 3). At |μ| = 2, where every σ is testable, W1 EFFECT holds in 0 of 20 testable runs (σ = 1), 10 of 20 (σ = 1.5) and
30 of 30 (σ = 3), so width matters at matched position, but the "both metrics" headline is a property of wide hot states and of which hot states pass P0. R and the direct sign agreed in 225 of 225 runs without excluding r ∈ [0.95, 1.05]; r < 1 in all 225, so this sweep also
does not probe r > 1.

**Other observables (descriptive).** At κ = 0 the cold state of this note sits exactly at the equilibrium value of ⟨U⟩ (its distance is 3·10⁻¹⁷, round-off), so the energy comparison is degenerate there: "energy shows no crossing" cannot be tested with these states at
κ = 0. At κ ≠ 0, ⟨U⟩ crosses at 216 of 216 points and Var(x) at 288 of 288, but the near-degeneracy recurs, just weaker, at the smallest tilt: at κ = 0.02 the P0 ratio D_hot(0)/D_cold(0) for ⟨U⟩ is 5.8·10³–1.4·10⁴ (`u_p0`, `summary_main.csv`) —
orders of magnitude above the 1.25 threshold, which here means the cold state's ⟨U⟩ starts essentially at its own equilibrium value (a small-κ echo of the exact κ = 0 degeneracy above), not that the test is close to failing. No P0-style filter was applied
to ⟨U⟩ or Var(x) in this run, so the 216/216 and 288/288 counts include points where the comparison is close to degenerate in this sense.

## 5. Case study: the earlier analysis

**The earlier metric.** For each observable O the earlier analysis took O_eq as the mean of the last 10% of the two compared trajectories, computed gap_v1 = D_hot − D_cold along the run (the sign convention is opposite to Section 3) and reported the fraction of recorded time
with gap_v1 < 0 ("% crossed"). The earlier draft's own prose (§II.D) states the required initial-ordering check; its code (`run_phase_diagram.py`) does not implement it — `crossed_pct` is computed directly from `gap`, with no test on the
initial distances. For p_left the equilibrium is 0.5 at κ = 0 and the earlier grid had κ = 0 only; a cold state fully in one well starts at the maximal distance 0.5, so with the analytic equilibrium the initial-ordering condition is
unsatisfiable for any hot state (0 of 72 points; with the tail-estimated equilibrium it fails for this hot state, not structurally). With tilt this argument no longer holds.

**Ablation on noise-free numerical trajectories (72 points, κ = 0, the earlier initial states, x marginal; ranges over four implicit-Euler time steps h = 0.05, 0.025, 0.0125, 0.00625 per recorded interval of 0.25):**

| Variant | Points with "crossing" > 70% of the recorded time | Mean over points |
|---|---|---|
| earlier metric (tail-estimated O_eq, no P0 check) | 57–62 / 72 | 82.4–88.3% |
| analytic O_eq = 0.5, still no P0 check | 70–72 / 72 | 98.1–100.0% (minimum 25.5–99.3%) |
| either estimate + P0 check | none of the 72 points can be tested | — |

With the analytic equilibrium the symmetric hot state has D_hot ≡ 0 for p_left, so it is "closer" from the first record: row 2 states a property of the design, not a discovery. For the earlier metric the first record already has gap_v1 < 0 at 69–71 of the 72 points (646
of 720 runs originally). The tail-estimated equilibrium is not necessary and lowers the fraction. Its bias is real: mean deviation 0.095 (maximum 0.25) from 0.5, consistent with 0.097 and 0.259 in the original simulations. At the earlier reference point (b = 1, T = 0.2, a
non-degenerate point) the noise-free metric gives 95.0% against 94.8 ± 0.6% in the earlier ten-seed runs. The statistics fluctuate non-monotonically with the time step (Pearson 0.25 / 0.33 / 0.22 / 0.29 and mean discrepancy 25.4 / 24.7 / 26.2 / 24.9 points for the four steps; at the
one point that drives the minimum of row 2 the hot-closer fraction drops to 25.5% at h = 0.0125 only): the metric is decided by differences at the level of discretisation error at some points, so its noise-free values carry a spread of a few points. The ablation explains why a
crossing was reported; it does not explain the shape of the earlier map, nor the earlier energy claim.

**The earlier phase map (Pearson 0.22–0.33; 57–62 versus 27 points above 70%; mean discrepancy 24.7–26.2 points, worse than the constant predictor 18.5).** The quantities below are for h = 0.025. *(i) Finite sample and gradient clipping (attribution test defined before it was coded).*
Rerun of the earlier process (Euler–Maruyama step η = 0.005, 30 000 steps, N = 5000, with the gradient clipped to ±100 and the position to ±50 as in the earlier code) at four points chosen by a preset rule (the two largest and two smallest discrepancies), with N = 5000 and 50 000 and with and
without clipping, five (N = 5000) or two (N = 50 000) seeds. Finite sample was supported at 1 of 4 points and clipping at 0 of 4 (threshold 3 of 4), so neither is supported; but the test could barely pass as specified, since two of the four points have almost no discrepancy to close, and "neither" is a
result of the preset rule, not evidence against finite sample. (A first version used overlapping noise streams across seeds; the result was unchanged after fixing it. Without clipping at b = 5 up to 2.5% of particles diverge.) At the extreme point (b = 2, T = 0.05) N = 5000 gives 20.1 ± 44.6% (the earlier map: 20.0 ± 42.1) and N = 50 000
gives 99.7% in both seeds, i.e. the bimodality collapses toward a deterministic value; at (b = 1.5, T = 0.05) it remains bimodal at N = 50 000 (two runs at 0.2% and 99.8%). *(ii) Degeneracy hypothesis (formulated after (i); its predictions were written into the analysis script before it ran but are not in the committed pre-specification).*
With the equilibrium from the two tails, gap_v1 is (nearly) zero when both trajectories are stationary throughout the run (cold stuck near 0, hot at 0.5), so the metric returns the sign of noise or of round-off. At the 11 points where the noise-free max|gap_v1| < 10⁻⁶ the earlier ten-seed standard deviation has median 51.4 points (mean 48.7) against a median 4.7 elsewhere (mean 7.7) and 100% of the individual
earlier runs there lie at an extreme (< 10% or > 90%) — as predicted; the prediction that reproduction would be good away from those points failed (mean discrepancy there 19.8). Four points just above the 10⁻⁶ threshold are also bimodal in this sense (SD 40.5–46.9:
(b,T) = (1.5, 0.1), (0.7, 0.05), (5, 0.3), (3, 0.2)), so the threshold does not mark a sharp boundary. At four of the 11 degenerate points max|gap_v1| ≈ 10⁻¹⁵, i.e. the noise-free value is decided by round-off; replacing those four by 50% changes the discrepancy statistics to Pearson 0.39, mean 22.6, 56 points above 70%.
*(iii) Post-hoc noise-floor model (formulated after (ii) partly failed).* Records with |gap_v1| ≤ θ are counted as 50% (in expectation; in single runs the sign is largely fixed through the tail-estimated equilibrium) and the others deterministically, with θ = 0.02 from the sampling noise of the difference of two fractions of 5000 particles (≈ 2√2·√(0.25/5000)), not fitted. It gives a mean
discrepancy of 8.8 points (Pearson 0.71; 6.9–9.8 for θ from 0.005 to 0.03) against baselines of 18.5 (the map's mean everywhere), 19.1 (50% at the 11 degenerate points, noise-free metric elsewhere) and 9.9 for a shrinkage baseline, (1 − f)·noise-free + 50·f with f the fraction of records with |gap_v1| ≤ 0.02. Stratified: at the 11 degenerate points
15.4 for the model and the two hybrid baselines (target noise floor 12.3); at the other 61 points 7.6 for the model and 8.9 for shrinkage, against a noise floor of 1.9. At 57 of 72 points more than half of the records have |gap_v1| ≤ 0.02 (this set overlaps the set of 57 points above 70% in 42 points). The model is therefore a descriptive fit with little advantage over shrinkage and about four times the noise floor at
non-degenerate points. *Out-of-sample check (θ scaled to N = 50 000, 0.0063):* at the four rerun points the predictions and observations are 50 versus 99.7 (b = 2, T = 0.05, a miss of 50 points), 50 versus 50.0 ± 70 (agreement only in the mean), 94.9 versus 95.0 (b = 1.5, T = 0.3) and 86.2 versus 95.1 (b = 5, T = 0.5; at N = 5000 the model gives 62 against 95 in both the earlier runs and the noise-free metric).
We therefore treat (ii) and (iii) as candidate mechanisms and do not claim a validated explanation of the earlier map.

**Other unsupported claims of the earlier draft.** (a) "No crossing in energy or position variance": with the cold state of this note the κ = 0 energy comparison is degenerate (Section 4), so nothing about energy follows from this note's κ = 0 results; with the earlier cold state N(1, 0.05²) and different y distributions we did not assess the claim beyond
reproducing the earlier count of points above 70% for energy (5 of 72); the earlier ⟨x²⟩ and the Var(x) used here are different observables. (b) A negative result for neural networks was presented as consistent with the mechanism; those experiments were invalid, so the result is "not tested"; this note makes no statement about neural networks. (c) The controls of the earlier draft
(balanced cold state, single-well potential, no gradient clipping, 30 seeds) each behaved as the mechanism predicted, but none tested whether the pipeline reports a crossing when the hot state simply starts closer.

## 6. Limitations

One-dimensional (x-marginal) reduction; two families of hot and cold states; a single potential family; Markovian overdamped dynamics; KL and W1 as headline metrics. The frozen pre-specification also lists total variation (TV) as a checking metric alongside W1
(`PREREG_v2.md`); it was never computed — an undisclosed departure from the plan, not a deliberate choice, corrected here (`PREREG_v2.md` D2.11). The hot-state sweep at b/T = 10 only and never with r > 1 except in G5b. Thresholds are preset heuristics (Section 3); P0 was missed by 0.001 at five points. Two grid levels cannot confirm the convergence order; a dx = 0.0025
check at five points is reported in Section 4. For b/T > 12 (64 points) λ₁ is a first-passage estimate validated only where an eigenvalue computation is possible, and R was not evaluated (at κ ≠ 0 at 48 of those points). The cross-check against simulation used three points at κ = 0. The agreement of R with the direct calculation is a consistency check. The reproduction of the earlier metric depends on the
numerical time step at the level of a few points. The pre-specification is local, from one session, and its author is the assistant that also ran it. The code was reviewed by AI reviewers reading the code (verdicts on the solver and on the analysis code: LGTM after fixes; a later review of the post-hoc analysis scripts returned NEEDS_WORK, and its findings are addressed above where they affect quoted numbers); the reviewers could not always run their own probes; there is
no independent human replication.

## 7. Conclusion

In this model, on the tested grid (b/T from 0.1 to 100, κ up to 0.1) and with the tested hot-state family (a wide Gaussian passing rule H, or the σ = 3 swept states), a symmetric hot state crosses a one-well cold state wherever the precondition holds. At κ = 0 this is because its slowest-mode coefficient vanishes by symmetry (the strong Mpemba effect in the sense of Klich et al.); at κ ≠ 0 up to 0.1 it is because the overlap with the slowest mode is small (r ≤ 0.43 where evaluated). The result is largely determined by the design and is not new.
The earlier reported crossing was produced by a metric with no check that the hot state starts farther, which by itself yields a "crossing" for p_left at κ = 0; the earlier phase map remains only qualitatively understood (degeneracy and sub-noise sign flips are candidate mechanisms, not validated ones); and the earlier energy claim was not assessed with its own states. Checks that follow from this one case: verify the initial
ordering (P0); use an analytic or independently converged equilibrium; ask whether the observable's distance can exceed the cold state's initial distance; test for degeneracy when the cold state already sits at the observable's equilibrium value; check that a metric evaluated on noise-free data is stable under the time step; and include a control the metric must fail.

## Data and code availability

Repository: https://github.com/sergeeey/infompemba-langevin-crossing (branch `feature/prereg-v2`, tag `prereg-v2-frozen`). [Archive DOI to be supplied by the author once minted; see `ZENODO_SETUP.md` in the repository.] Per-configuration summary: `prereg_v2/summary_main.csv`; manifest with SHA-256 of the 288 curve files: `prereg_v2/out_manifest.csv`; sweep: `prereg_v2/summary_robustness.csv`; comparison with the earlier map: `prereg_v2/v1_compare_points.csv`; the four time-step
variants of the reproduction underlying the ranges in Section 5: `prereg_v2/cmp_sub{5,10,20,40}.csv`. **Not deposited:** the 288 per-configuration curve files (`prereg_v2/out/*.npz`, about 42 MB, listed by hash in `out_manifest.csv` but excluded
from the repository by size) — the main-run numbers in Sections 4–5 that read directly from these files (e.g. the margin statistics) cannot be regenerated from the deposit alone without rerunning `run_prereg_v2_main.py`.

## References (identifiers verified in OpenAlex/Crossref; see `prereg_v2/N1_LITERATURE.md`)

- Z. Lu and O. Raz, Proc. Natl. Acad. Sci. USA 114, 5083 (2017), doi:10.1073/pnas.1701264114.
- I. Klich, O. Raz, O. Hirschberg and M. Vucelja, "Mpemba index and anomalous relaxation", Phys. Rev. X 9, 021060 (2019), doi:10.1103/physrevx.9.021060.
- A. Kumar and J. Bechhoefer, "Exponentially faster cooling in a colloidal system", Nature 584, 64 (2020), doi:10.1038/s41586-020-2560-x.
- R. Chétrite, A. Kumar and J. Bechhoefer, "The metastable Mpemba effect corresponds to a non-monotonic temperature dependence of extractable work", Front. Phys. (2021), doi:10.3389/fphy.2021.654271.
- T. V. Vu and H. Hayakawa, "Thermomajorization Mpemba effect", Phys. Rev. Lett. 134, 107101 (2025), doi:10.1103/physrevlett.134.107101.
- A. Biswas, R. Rajesh and A. K. Pal, "Mpemba effect in a Langevin system: Population statistics, metastability, and other exact results", J. Chem. Phys. 159 (4) (2023) [article number to be added], doi:10.1063/5.0155855.
- A. Biswas and R. Rajesh, "Mpemba effect for a Brownian particle trapped in a single well potential", Phys. Rev. E 108, 024131 (2023), doi:10.1103/physreve.108.024131.
- A. Biswas, V. V. Prasad and R. Rajesh, "Mpemba effect in driven granular gases: role of distance measures", arXiv:2303.10900 (2023).
- H. Hayakawa and S. Takada, "Mpemba effect in a two-dimensional bistable potential", arXiv:2603.24148 (2026).
- G. Teza et al., "Speedups in nonequilibrium thermal relaxation: Mpemba and related effects", Phys. Rep. (2025), doi:10.1016/j.physrep.2025.10.009.
- D. L. Scharfetter and H. K. Gummel, IEEE Trans. Electron Devices 16, 64 (1969), doi:10.1109/t-ed.1969.16566 [DOI verified; authors and volume from memory].
- W. K. Grassmann, M. I. Taksar and D. P. Heyman, Oper. Res. 33, 1107 (1985), doi:10.1287/opre.33.5.1107 [title and DOI verified; authors from memory].
- [Other references of the earlier draft are to be re-verified by the author before use; item 9 there has an erratum, Phys. Rev. Lett. 128, 229901 (2022); item 4 (Mpemba & Osborne 1969) could not be verified; item 11 was verified on 2026-09-29 as arXiv:2507.04206 (S. Liu and Z. Hu, 2025; the earlier draft's initial "J." was wrong; abstract page checked, full text not read).]

## Appendix: timeline of the pre-specification (local git, one machine; the repository was pushed and made public only after this timeline was complete, on 2026-09-29)

`3e002b6` baseline; `db3de4a` tag `prereg-v2-frozen` (criteria, P0, K0–K4); `2ac799e` solver-method deviation and validation tolerances (before the solver code); `ce807a5` solver and validation gates; `7d71290`, `dc145c9` main-run protocol, controls and spectral criterion (before the corresponding code and runs); `bde4119` (sweep), `6c96eb9`
(attribution, W1 control, equivalence check), `52b6d7b` (margin correction and refinement check), each before its code. Tolerances for the simulation cross-check were written into the file before its code but committed together with it; the degeneracy predictions, the noise-floor model, the baselines and the time-step check were written into analysis scripts or results files, not into
the pre-specification. Post-hoc additions are listed in `PREREG_v2.md` D2.7–D2.10 and `prereg_v2/RESULTS.md` §8–§10.
