# Observable-specific Mpemba effect in double-well Langevin dynamics: role of metastable basin imbalance

Sergei Boiko$^{1}$

$^{1}$ Independent researcher, Almaty, Kazakhstan

---

## Abstract

We demonstrate a robust Mpemba-like anomalous relaxation in overdamped Langevin
dynamics with a double-well potential U(x,y) = b(x^2 - 1)^2 + 0.3 y^2. In a
systematic study of 720 independent simulations across 9 barrier heights and 8
temperatures, we find that the effect is **observable-specific**: a "hot" initial
condition (spread across both wells) relaxes to equilibrium basin proportions
faster than a "cold" condition (trapped in one well) for the basin-occupancy
observable p_left, while no crossing occurs in energy or position variance. The
effect is strong (>90% of trajectories cross) in a well-defined region of
parameter space (b = 0.5-2.0, T = 0.1-0.2), with a peak of 95.4% at b = 2.0,
T = 0.2. Four independent controls confirm the mechanism: balanced-cold
initialization eliminates the effect, single-well potential shows no effect,
results are unchanged without gradient clipping, and 30 random seeds all produce
strong crossings. Spectral analysis of the Fokker-Planck operator reveals that
the Mpemba region coincides with large timescale separation between inter-basin
(slow) and intra-basin (fast) relaxation modes. The Kramers escape rate formula
predicts the slow eigenvalue with 89% accuracy (within factor 2). We argue that
the observable-specificity arises because basin occupancy projects onto the slow
inter-basin mode, while energy-like observables project predominantly onto fast
intra-basin modes. Our results suggest that previous negative reports of the
Mpemba effect may reflect the choice of observable rather than the absence of the
phenomenon.

---

## I. Introduction

The Mpemba effect --- the counterintuitive observation that a hotter system can
sometimes relax to equilibrium faster than a cooler one --- has been a subject of
renewed scientific interest following its rigorous formulation in Markovian
dynamics [1-3]. Originally observed in water freezing [4], analogues have since
been demonstrated in spin systems [5], colloidal particles [6], granular gases
[7], and clathrate hydrates [8].

Recent theoretical work has established that the Mpemba effect in Markov
processes is governed by the spectral structure of the transition operator [1,2].
Specifically, if the slowest relaxation mode has an anomalous initial overlap
with the hot distribution compared to the cold one, faster relaxation can occur.
Lu and Raz [1] provided a general spectral criterion, and subsequent work has
explored strong vs. weak forms of the effect [3,9]. A recent thermomajorization
framework [10] proves that when initial states satisfy a thermomajorization
ordering, the Mpemba effect holds for all monotone distance measures
simultaneously --- eliminating observable dependence entirely. Most recently,
Summer et al. [14] provided a resource-theoretical unification of classical and
quantum Mpemba effects, showing that both arise from the same underlying logic
of athermality as a resource.

Despite this theoretical progress, a key question remains underexplored: **what
happens when thermomajorization does not hold?** In that regime, different
observables may couple to different relaxation modes, and the Mpemba effect
should depend on which mode dominates the observable of interest. Most studies
define relaxation in terms of a global distance measure (total variation, KL
divergence, or energy), leaving the observable-specific regime uncharted.

In this work, we address this question directly. We study overdamped Langevin
dynamics in a 2D double-well potential and show that:

1. A robust Mpemba-like crossing occurs in the basin-occupancy observable
   p_left, with >90% of trajectories crossing in the optimal parameter regime.

2. **No crossing occurs** in energy or position-variance observables for the same
   trajectories and parameters.

3. The mechanism is metastable basin imbalance: the hot distribution starts
   closer to the equilibrium basin proportions, while the cold distribution is
   trapped in one well.

4. The phase boundary of the effect is predicted by the timescale separation
   between the inter-basin and intra-basin relaxation modes of the Fokker-Planck
   operator.

These results demonstrate that the Mpemba effect is not a property of the system
alone, but of the system-observable pair. This observable-specificity has direct
implications for experimental design: searching for the Mpemba effect with the
wrong observable may yield a false negative.

---

## II. Model and Methods

### A. System

We consider overdamped Langevin dynamics in 2D:

    dx = -nabla U(x,y) dt + sqrt(2T) dW

with the double-well potential

    U(x,y) = b (x^2 - 1)^2 + 0.3 y^2

where b > 0 is the barrier height and T > 0 is the temperature. The potential has
two minima at (x,y) = (+/-1, 0) separated by a barrier of height b at the origin.

### B. Initial conditions

We define two classes of initial conditions:

- **Cold**: particles sampled from N((1, 0), sigma_c^2 I) with sigma_c = 0.05.
  All particles start in the right well. The cold distribution is far from
  equilibrium in basin occupancy (p_left approx 0) but close to equilibrium
  within the occupied well.

- **Hot**: particles sampled from N((0, 0), sigma_h^2 I) with sigma_h = 2.0.
  Particles are spread across both wells approximately equally. The hot
  distribution is closer to equilibrium basin proportions but far from
  equilibrium in energy.

### C. Observables

We track three observables:

1. **Basin occupancy**: p_left(t) = fraction of particles with x < 0.
2. **Mean energy**: <E(t)> = <U(x(t), y(t))>.
3. **Position variance**: <x(t)^2>.

For each observable O(t), we define the distance to equilibrium as
|O(t) - O_eq|, where O_eq is estimated as the average of the final 10% of both
trajectories.

### D. Mpemba crossing criterion

A Mpemba-like crossing is detected when the hot trajectory's distance to
equilibrium drops below the cold trajectory's distance:

    |O_hot(t_c) - O_eq| < |O_cold(t_c) - O_eq|

for some time t_c, with the initial ordering reversed:

    |O_hot(0) - O_eq| > |O_cold(0) - O_eq|.

We quantify the effect by the fraction of simulation time during which the hot
trajectory is closer to equilibrium ("% crossed").

### E. Simulation parameters

Each simulation uses N = 5000 particles with step size eta = 0.005 and
n_steps = 30000 (evaluated every 50 steps). For the phase diagram, we sweep
b in {0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0} and T in
{0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0}, with 10 independent random seeds
per point (720 runs total). Robustness is confirmed with 30 seeds at the
reference point (b=1.0, T=0.2).

### F. Spectral analysis

The Fokker-Planck operator for the 1D marginal (integrating out y) is
transformed to Schrodinger form via the similarity transformation
H = exp(U/2T) L exp(-U/2T), yielding a real symmetric operator with eigenvalues
0 = lambda_0 > -lambda_1 > -lambda_2 > ... The rates lambda_k give the
relaxation timescales tau_k = 1/lambda_k. We compute these numerically via
sparse eigenvalue decomposition (shift-invert mode) on a grid of 501 points, and
compare with the Kramers escape rate formula for lambda_1.

---

## III. Results

### A. Observable-specific crossing

Figure 1 shows the central result. At the reference parameters (b = 1.0,
T = 0.2), the distance to equilibrium for p_left shows a clear crossing: the hot
trajectory relaxes faster and remains closer to equilibrium for >94% of the
simulation. In contrast, the energy and position-variance observables show no
crossing --- the hot trajectory remains farther from equilibrium throughout.

This observable-specificity is the key finding. The same trajectories, same
dynamics, same initial conditions produce opposite conclusions depending on which
observable is measured.

### B. Mechanism: metastable basin imbalance

Three control experiments (Fig. 3) confirm the mechanism:

**(i) Balanced cold.** When the cold initial condition is distributed equally
across both wells (50% left, 50% right), the crossing fraction drops to 28%
(consistent with noise). The effect requires basin imbalance.

**(ii) Single-well potential.** In a harmonic potential U = x^2 + y^2 (no
metastability), the crossing fraction is 47% (noise). The effect requires
metastable basins.

**(iii) Clipping control.** Running identical simulations without gradient
clipping yields 94.8% vs. 94.6% crossing, confirming the effect is not a
numerical artifact.

**(iv) Seed robustness.** All 30 of 30 random seeds produce "strong" crossing
(>70% of time crossed) at the reference parameters. Mean crossing fraction:
94.7%.

The mechanism is clear: the hot distribution starts with approximately equal
occupation of both wells, close to the equilibrium ratio. The cold distribution
is entirely in one well and must wait for the slow barrier-crossing process to
redistribute. Basin occupancy measures exactly this slow redistribution, which is
why it shows the crossing.

### C. Phase diagram

Figure 2(a) shows the full phase diagram. The effect is strong (>90% crossed) in
a connected region spanning b = 0.5--2.0 and T = 0.1--0.2, with a peak of 95.4%
at b = 2.0, T = 0.2. The effect vanishes in two regimes:

- **High temperature** (T > 0.5): thermal fluctuations wash out the basin
  imbalance before it can create a measurable crossing.
- **High barrier + low temperature** (b > 3, T < 0.1): the barrier is
  effectively impassable --- both hot and cold distributions remain trapped,
  and the slow mode becomes inaccessible on simulation timescales.

25 of the 72 parameter combinations show strong effects (>=70% of seeds with
>70% crossing).

### D. Spectral prediction

Figure 2(b) shows the timescale ratio tau_1/tau_2 from the Fokker-Planck
spectrum. Large timescale separation (tau_1 >> tau_2) means that the inter-basin
relaxation is much slower than intra-basin relaxation --- the necessary condition
for the basin-imbalance mechanism.

The spectral criterion tau_1/tau_2 > 10 captures all 25 strong Mpemba points
(recall = 100%). Its precision is 45%: some points with large timescale
separation do not show Mpemba because the barrier is effectively impassable. This
is physically correct --- timescale separation is a **necessary** condition (the
slow mode must exist and dominate), but not sufficient (the mode must also be
**accessible** within the simulation time). In the high-barrier regime,
tau_1 diverges exponentially while particle mobility vanishes, preventing the
redistribution needed for the crossing.

The Kramers escape rate formula predicts lambda_1 with a median accuracy ratio of
1.04, and 89% of predictions fall within a factor of 2 of the numerical value
(Fig. 4). Crucially, the spectral analysis is not a post-hoc fit to the Mpemba
data: it uses only the potential U(x) and temperature T as inputs, with no
adjustable parameters and no reference to the crossing statistics. The Kramers
formula was derived in 1940 [8], decades before the Mpemba effect was formalized
in Markov systems. Its success here represents an independent theoretical
prediction, not a retrospective explanation.

### E. Observable-specificity explained

The spectral decomposition explains why p_left shows the crossing while energy
does not. The eigenmodes of the Fokker-Planck operator (Fig. 4, top) reveal:

- psi_1 (slow, inter-basin mode): antisymmetric, one lobe in each well. This
  mode controls the redistribution of probability between basins.
- psi_2 (fast, intra-basin mode): symmetric, localized within each well. This
  mode controls relaxation of the local distribution shape.

Basin occupancy p_left projects strongly onto psi_1 (it measures the left-right
asymmetry). Energy projects primarily onto psi_2 and higher modes (it measures
the shape of the distribution within wells, not which well particles occupy). The
Mpemba crossing occurs because the hot distribution has a smaller overlap with
psi_1 than the cold distribution --- but this advantage is invisible to
observables that couple to faster modes.

One might argue that this observable-specificity is trivially expected given
that p_left is "designed" to be sensitive to basin occupation. However, the key
point is not the individual result but the general principle: for any
metastable system, the spectral decomposition provides an a priori criterion
for which observables will show Mpemba-like behavior (those projecting onto the
slowest mode) and which will not. Without this analysis, there is no way to know
in advance --- and an experimenter measuring only energy would conclude the
effect is absent.

---

## IV. Discussion

### Relation to spectral theory and thermomajorization

Our results provide a concrete realization of the spectral Mpemba criterion of
Lu and Raz [1], with the added insight of observable-specificity. Their framework
predicts that anomalous relaxation occurs when the initial distribution's
projection onto the slowest mode is anomalously small. We show that this
projection matters only for observables that couple to that mode --- a point that
is implicit in the theory but has not been explicitly demonstrated.

Our system also provides a concrete example of the boundary of the
thermomajorization framework [10]. Vu and Hayakawa proved that when initial
states satisfy thermomajorization ordering, Mpemba holds for all monotone
measures. Our cold and hot initial conditions evidently do not satisfy this
ordering: basin occupancy shows Mpemba while energy does not. The eigenmode
projection mechanism we identify explains precisely why universality fails here
--- the two observables couple to different spectral sectors of the
Fokker-Planck operator. In the language of Summer et al. [14], our hot initial
condition has lower "athermality" in the basin-occupation sector but higher
athermality in the energy sector, providing a concrete continuous-space example
of the resource-theoretical hierarchy they establish for discrete systems.

### Timescale separation as necessary but not sufficient

The spectral prediction achieves perfect recall (100%) but limited precision
(45%). This asymmetry has a clear physical interpretation: large tau_1/tau_2
means the slow mode exists and is well-separated, but if tau_1 itself is
astronomically large (exponentially in b/T), the redistribution simply does not
occur on accessible timescales. The Mpemba effect thus requires a "Goldilocks
zone" of metastability: barriers high enough to create distinct basins, but low
enough for thermally-activated crossing.

### Implications for Mpemba searches

Our finding that the same system shows a strong Mpemba effect in one observable
and none in another has practical implications. Experimental studies that
measure only energy-like quantities (heat release, temperature) may miss a
Mpemba effect that is present in other observables (magnetization in spin
systems, phase fractions in mixtures). Conversely, claims of "no Mpemba effect"
should specify which observable was measured.

### Neural network training (negative transfer)

We also tested whether the basin-imbalance mechanism transfers to neural network
training (MLP on FashionMNIST), using KL divergence to a reference model as the
observable. Five experimental configurations showed no crossing. The root cause
is twofold: (i) the MLP weight space has no discrete metastable basins (verified
by clustering analysis showing pairwise cosine distances of 1.00 +/- 0.01), and
(ii) the "cold" initial condition was not stationary (KL drift of +7400%). This
negative result is consistent with our mechanism: without metastable basins, the
basin-imbalance pathway does not exist. Details are in the Supplemental Material.

We emphasize that this negative result is specific to the basin-imbalance
mechanism in raw weight space. It does not rule out Mpemba effects in neural
networks via other pathways: Liu and Hu [11] recently demonstrated Mpemba-like
behavior in large language model training through a continuous spectral mechanism
in a valley-river loss landscape, confirming that the effect can arise in ML
without discrete basins. The combination of our positive toy result, our negative
ML transfer, and their positive LLM result paints a coherent picture: the Mpemba
effect requires spectral structure (a slow mode dominating relaxation), but the
specific realization --- discrete basins vs. continuous landscape geometry ---
varies across systems. Our contribution is identifying observable-specificity as
a key diagnostic, applicable regardless of the underlying mechanism.

### Limitations

1. Our system is 2D with a specific potential. While the phase diagram
   demonstrates generality across parameters, extension to higher dimensions and
   other potential forms remains open.
2. The spectral analysis uses the 1D marginal. The full 2D spectrum may reveal
   additional structure.
3. The simulations use finite time and particle number. Very rare barrier
   crossings at high b/low T may be undersampled.

---

## V. Conclusion

We have demonstrated that the Mpemba effect in a double-well Langevin system is
observable-specific: it appears robustly in basin occupancy but not in energy or
position variance. The mechanism is metastable basin imbalance, and the phase
boundary is predicted by the Fokker-Planck spectral gap. The choice of
observable determines whether the Mpemba effect is detectable.

---

## References

[1] Z. Lu and O. Raz, "Nonequilibrium thermodynamics of the Markovian Mpemba
    effect and its inverse," PNAS 114, 5083 (2017).

[2] I. Klich, O. Raz, O. Hirschberg, and M. Vucelja, "Mpemba index and
    anomalous relaxation," Phys. Rev. X 9, 021060 (2019).

[3] A. Kumar and J. Bechhoefer, "Exponentially faster cooling in a colloidal
    system," Nature 584, 64 (2020).

[4] E. B. Mpemba and D. G. Osborne, "Cool?," Phys. Educ. 4, 172 (1969).

[5] A. Lasanta, F. Vega Reyes, A. Prados, and A. Santos, "When the hotter
    cools quicker: Mpemba effect in granular fluids," Phys. Rev. Lett. 119,
    148001 (2017).

[6] A. Gal and O. Raz, "Precooling strategy allows exponentially faster
    heating," Phys. Rev. Lett. 124, 060602 (2020).

[7] M. Baity-Jesi et al., "The Mpemba effect in spin glasses is a persistent
    memory effect," PNAS 116, 15350 (2019).

[8] H. A. Kramers, "Brownian motion in a field of force and the diffusion model
    of chemical reactions," Physica 7, 284 (1940).

[9] A. Lapolla and A. Godec, "Faster uphill relaxation in thermodynamically
    equidistant temperature quenches," Phys. Rev. Lett. 125, 110602 (2020).
    <!-- Note: replaces erroneous Manikandan PRL 127,180603 which was a different
    paper. Lapolla & Godec is the correct classical strong/weak forms ref.
    Manikandan PRResearch 3, 043108 (2021) covers quantum few-level case,
    not needed for our classical paper. -->

[10] T. V. Vu and H. Hayakawa, "Thermomajorization Mpemba effect," Phys. Rev.
     Lett. 134, 107101 (2025).

[11] J. Liu and Z. Hu, "Mpemba effect in large language model training
     dynamics," arXiv:2507.04206 (2025).

[12] G. Teza, J. Bechhoefer, et al., "Speedups in nonequilibrium thermal
     relaxation: a review of the Mpemba and related effects," arXiv:2502.01758
     (2025).

[13] P. Chattopadhyay, J. F. G. Santos, and A. Misra, "Anomaly to resource:
     the Mpemba effect in quantum thermometry," arXiv:2601.05046 (2026).

[14] A. Summer, M. Moroder, L. P. Bettmann, X. Turkeshi, I. Marvian, and
     J. Goold, "Resource-theoretical unification of Mpemba effects: classical
     and quantum," Phys. Rev. X 16, 011065 (2026). DOI: 10.1103/rbt4-psfd.

---

## Figure Captions

**Figure 1.** Observable-specific Mpemba effect. Distance to equilibrium
|O(t) - O_eq| for cold (blue) and hot (red) initial conditions, averaged over
10 seeds. (a) Basin occupancy p_left shows a clear crossing: the hot trajectory
relaxes faster. (b) Energy shows no crossing. (c) Position variance x^2 shows no
crossing. Parameters: b = 1.0, T = 0.2, N = 5000 particles. Shaded regions:
+/- 1 standard deviation.

**Figure 2.** Phase diagram and spectral prediction. (a) Mean % of simulation
time with hot closer to equilibrium (p_left), across 9 barrier heights and 8
temperatures (10 seeds each). Green = strong Mpemba, red = no effect.
(b) log_10(tau_1/tau_2) from the Fokker-Planck spectrum. (c) Overlay: numerical
Mpemba (color) with spectral contours (blue lines) at tau_1/tau_2 = 5, 10, 50.

**Figure 3.** Mechanism controls. (a) Unbalanced cold (one well): 100% crossing
--- strong effect. (b) Balanced cold (both wells): 28% --- effect eliminated.
(c) Single-well (harmonic): 47% --- no effect without metastability. All panels
show |p_left - p_eq| vs step, 10 seeds.

**Figure 4.** Spectral analysis. Top: Fokker-Planck eigenmodes psi_0
(stationary), psi_1 (inter-basin, slow), psi_2 (intra-basin, fast) at three
parameter points. Bottom left: Kramers formula vs numerical lambda_1 (94%
within factor 2). Bottom right: timescale ratio tau_1/tau_2 vs numerical %
crossed, confirming that large separation is necessary for the effect.
