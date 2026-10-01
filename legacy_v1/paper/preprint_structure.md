# Observable-specific Mpemba effect in double-well Langevin dynamics: role of metastable basin imbalance

## Target venue
Physical Review E — Rapid Communication (4 pages + supplemental)
Alternative: J. Stat. Mech. (Letter)

## Authors
Sergei Boiko (Independent researcher, Almaty, Kazakhstan)

---

## Abstract (~150 words)

We report a robust Mpemba-like anomalous relaxation in overdamped Langevin dynamics
with a double-well potential. The key finding is that the effect is
**observable-specific**: it appears in basin occupancy (the fraction of particles in
a given well) but is absent in energy-like observables. We demonstrate that the
mechanism is driven by metastable basin imbalance — a "hot" initial condition
spread across both wells relaxes faster to equilibrium basin proportions than a
"cold" condition trapped in one well. We confirm this through four independent
controls: (i) balanced-cold initialization eliminates the effect, (ii) single-well
potential shows no effect, (iii) barrier and temperature sweeps reveal a sharp phase
boundary, and (iv) spectral analysis of the Fokker-Planck operator shows that the
effect coincides with strong timescale separation between inter-basin and intra-basin
relaxation modes. Kramers formula predicts the phase boundary with 89% accuracy.

---

## Figures (4 main + supplemental)

### Figure 1: Core result
- (a) Mean p_left vs time for hot (red) and cold (blue) trajectories.
      Shaded = 95% CI across 10 seeds. Clear crossing visible.
- (b) Same for mean energy — no crossing (flat comparison).
- (c) Same for mean x² — no crossing.
- Caption: "Observable-specific Mpemba effect. The hot trajectory crosses the cold
  in basin occupancy p_left (a) but not in energy (b) or position variance (c)."

### Figure 2: Phase diagram (barrier × T)
- (a) Heatmap: % crossed (numerical, 10 seeds per point)
- (b) Heatmap: log₁₀(τ₁/τ₂) from Fokker-Planck spectrum
- (c) Overlay: Mpemba boundary vs spectral contour τ₁/τ₂ = 10
- Caption: "Phase diagram and spectral prediction. Strong Mpemba region coincides
  with large timescale separation."

### Figure 3: Controls
- (a) Unbalanced cold: 94.7% crossed (30/30 seeds STRONG)
- (b) Balanced cold: 43.2% (random — effect killed)
- (c) Single-well: 50.2% (random — no metastability)
- (d) Clipping vs no-clipping: identical (not an artifact)
- Caption: "Kill criteria confirm basin imbalance mechanism."

### Figure 4: Spectral analysis
- 3×3 grid of eigenmodes at different (b, T) — already generated
- Kramers vs numerical λ₁ scatter
- Caption: "Fokker-Planck eigenmodes and Kramers formula validation."

### Supplemental figures
- S1: Full robustness data (30 seeds, distribution of crossing times)
- S2: Barrier sweep at fixed T, Temperature sweep at fixed barrier (1D slices)
- S3: ML negative result summary (1 page)

---

## Section outline

### I. Introduction (0.5 page)
- Mpemba effect: history (water, spin systems, colloidal particles)
- Recent theoretical frameworks: Markovian relaxation, spectral theory (Lu & Raz 2017)
- Open question: which observables show the effect?
- Our contribution: observable-specificity + mechanism identification

### II. Model and methods (0.5 page)
- Potential: U(x,y) = b(x²-1)² + 0.3y²
- Overdamped Langevin: dx = -∇U dt + √(2T) dW
- Initial conditions: cold (one well, σ=0.05) vs hot (spread, σ=2.0)
- Metrics: p_left, energy, x²
- Statistical protocol: N=5000 particles, 30000 steps, 10-30 seeds

### III. Results (1.5 pages)
#### A. Observable-specific crossing
- p_left shows crossing; energy and x² do not
- Crossing is sustained (>94% of timesteps)

#### B. Mechanism: basin imbalance
- Balanced cold kills effect → 43% (random)
- Single-well baseline → 50% (no metastability)
- Clipping control → identical results

#### C. Phase diagram
- 9 barriers × 8 temperatures → sharp boundary
- Strong effect region: b ∈ [0.5, 2.0], T ∈ [0.1, 0.2]
- Dead zones: too high T (thermal noise dominates), too high barrier (particles stuck)

#### D. Spectral prediction
- Fokker-Planck eigenvalue problem in Schrodinger form
- τ₁/τ₂ predicts Mpemba boundary
- Kramers formula: 89% within factor 2 — analytical prediction works
- Observable-specificity explained: p_left projects onto slow (inter-basin) mode,
  energy projects onto fast modes

### IV. Discussion (0.5 page)
- Relation to Lu & Raz (2017): our result is a concrete realization of their
  spectral criterion, with the addition of observable-specificity
- Why energy doesn't show Mpemba: energy relaxation dominated by intra-well
  modes (fast), which don't have basin asymmetry
- Implications: previous negative results may have measured the "wrong" observable
- ML negative transfer: briefly mention (detail in supplemental)
- Open: does this mechanism transfer to high-dimensional metastable systems?

### V. Conclusion (3 sentences)
- Observable-specific Mpemba confirmed in double-well Langevin
- Mechanism = basin imbalance, predicted by Fokker-Planck timescale separation
- The choice of observable determines whether Mpemba is detectable

---

## Claim (one sentence for the paper)

> In overdamped Langevin dynamics with a double-well potential, a Mpemba-like
> crossing occurs specifically in the basin-occupancy observable and is driven by
> metastable imbalance between initial conditions; the effect is absent in
> energy-like observables and requires tunable metastability, with its phase
> boundary accurately predicted by the Fokker-Planck spectral gap.

---

## Key references to cite
- Mpemba & Osborne (1969) — original observation
- Lu & Raz (2017) — spectral theory of Mpemba in Markov systems
- Kramers (1940) — escape rate theory
- Kumar & Bechhoefer (2020) — experimental Mpemba in colloidal systems
- Lasanta et al. (2017) — Mpemba in granular gases
- Klich, Raz, Hirschberg, Vucelja (2019) — Mpemba index

---

## Data and code availability
All code, data, and analysis notebooks at: [GitHub link TBD]
