# Supplemental Material: Neural Network Transfer (Negative Result)

## Setup

We tested whether the Mpemba-like crossing observed in the double-well toy model
transfers to neural network training. We trained MLP networks (784-256-128-10,
tanh activation) on FashionMNIST with batch size 16 (to ensure heavy-tailed
gradient noise).

## Initial conditions

| Name | Method | KL to theta* |
|------|--------|-------------|
| theta* (reference) | AdamW, 50 epochs, cosine annealing | 0.0 |
| theta_cold | theta* + N(0, 0.01^2) noise | 0.0024 |
| theta_hot | Kaiming(gain=3.0) + Laplace(0.5) | 14.34 |

## Experiments (5 configurations)

1. VanillaSGD lr=0.01: both trajectories diverge, KL gap 0.96
2. LangevinFisher lr=0.01, damping=1e-8: KL explosion (damping too small)
3. LangevinFisher lr=0.001, damping=1.0: KL gap 3.93, cold diverges
4. LangevinFisher lr=0.001, 8000 iter: KL gap 1.15, slope decelerating
5. LangevinFisher lr=0.0003, noise=0.3, 30000 iter: KL gap 2.06

**No crossing observed in any configuration.**

## Root cause analysis

### Cold stationarity test
KL divergence of theta_cold drifted from 0.018 to 1.34 (+7400%) over training.
The "cold" initial condition was not stationary — all experiments were invalid
because the cold trajectory was actively learning, not relaxing.

### Basin structure test
We trained 20 MLPs with different random seeds and analyzed weight-space structure:
- Pairwise cosine distance: 1.000 +/- 0.008 (nearly orthogonal)
- PCA: each component explains ~5.4% (no dominant directions)
- Best silhouette score: 0.159 (k=4) — far below 0.25 threshold
- DBSCAN: 0 clusters at all epsilon values

**Conclusion:** MLP+FashionMNIST weight space has no discrete metastable basins.
The basin-imbalance mechanism confirmed in the toy model does not apply.

## Why the transfer failed

| Toy model | Neural network |
|-----------|---------------|
| Well-defined equilibrium | No clear equilibrium state |
| Two discrete basins | Continuous, high-dimensional landscape |
| Overdamped Langevin (exact) | SGD approximates Langevin only weakly |
| Basin occupancy observable | KL to theta* does not measure basin occupancy |

## Lessons learned

1. Always verify stationarity of the "cold" initial condition before comparing trajectories
2. KL divergence to a reference point is the wrong metric for detecting basin-occupancy Mpemba
3. Weight-space analysis is insufficient due to permutation symmetries
4. The mechanism (basin imbalance) requires metastable basins — verify their existence first

## Future directions

- Permutation alignment (Git Re-Basin) before clustering may reveal hidden basin structure
- Function-space clustering (compare by logits, not weights) may show metastable structure
- Teacher-student setup with 2+ teachers creates an explicitly multi-modal landscape
- Architectures with built-in symmetries may have discrete basins
