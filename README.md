> **Read this first (2026-09-30).** The current work in this repository is a **correction note and re-analysis** of an earlier claim
> of a "Mpemba-like crossing" in a double-well Langevin model: [`paper/note_v2_draft.md`](paper/note_v2_draft.md) (draft v0.5, not for
> submission; not a new result — the mechanism is known). Start with [`REPRODUCE.md`](REPRODUCE.md) to check or rerun it
> (environment: `requirements-v2.txt`). The pre-specification is frozen at tag `prereg-v2-frozen`
> (Zenodo snapshot of the frozen tag: 10.5281/zenodo.23039440; concept DOI for all versions: 10.5281/zenodo.23039439). The public git history was rewritten once on 2026-09-29: see [`HISTORY_REWRITE.md`](HISTORY_REWRITE.md).
>
> The earlier draft (`paper/preprint*`, `results/`) claimed an effect that the audit showed to be an artefact of the detector, and the
> GPU/neural-network pipeline described below produced experiments that are **invalid** (their "cold" state was not stationary);
> they support no statement about neural networks. They are kept for the record only.

# InfoMpemba: Mpemba Effect in Neural Network Information Geometry

## Status: **WORKING PROTOTYPE** (not yet scientifically verified)

This is a **functional experimental pipeline** for testing the Mpemba effect hypothesis
in neural network training dynamics. The prototype runs on GPU and passes initialization
validation. **Full scientific verification by the ТЗ criteria is NOT YET complete.**

## What Works

| Component | Status |
|-----------|--------|
| GPU support (RTX 5070 Ti, sm_120) | Verified |
| theta_star / theta_cold / theta_hot initialization | Verified |
| KL(cold) < 0.5, KL(hot) > 5.0 thresholds | Met (0.0024, 14.34) |
| Training pipeline (VanillaSGD, LangevinFisher) | Working |
| Metric computation on GPU (KL, Fisher trace) | Working |
| Results saved to Parquet | Working |

## What's NOT Yet Verified

| Requirement | Status |
|-------------|--------|
| Full 1000-run statistical study | NOT RUN |
| Crossing detection with 50-step persistence | NOT TESTED |
| Welch t-test p < 0.01 | NOT TESTED |
| Batch size = 16 full run | NOT RUN (pilot pending) |
| Reproducible environment lock | Partial (nightly build needed for sm_120) |

## Quick Start

### Prerequisites
- NVIDIA GPU with CUDA 12.8+ support (sm_120 / Blackwell)
- Python 3.11+

### Install
```powershell
# RTX 5070 Ti / Blackwell requires nightly build:
pip install --pre --no-cache-dir torch torchvision torchaudio ^
    --index-url https://download.pytorch.org/whl/nightly/cu128

pip install -r requirements.txt
```

### Verify GPU
```powershell
python launch_full_experiment.py --check-gpu
```

### Launch Options

```powershell
# 1. Quick test (20 runs x 100 iter, ~10 min)
python launch_full_experiment.py --quick-test --no-confirm

# 2. Pilot validation (50 runs x 500 iter, batch=16, ~2 hours)
python launch_full_experiment.py --pilot

# 3. Full ТЗ-compliant run (1000 runs x 2000 iter, batch=16, ~12-15 hours)
python launch_full_experiment.py
```

### Analysis
```powershell
jupyter notebook analysis/Mpemba_Crossing_Analysis.ipynb
```

## Project Structure

```
/InfoMpemba
├── src/
│   ├── initializers.py       # theta_star, theta_cold, theta_hot (+ auto-tuning)
│   ├── langevin_sgd.py       # VanillaSGD + Langevin-Fisher optimizers
│   ├── fisher_metrics.py     # KL divergence + Fisher trace (GPU-only)
│   └── mpemba_runner.py      # Statistical controller (N runs)
├── configs/
│   ├── full_tz.yaml          # Full ТЗ-compliant (batch=16, 1000 runs)
│   ├── pilot.yaml            # Pilot study (batch=16, 50 runs)
│   └── quick_test.yaml       # Auto-generated quick test
├── analysis/
│   └── Mpemba_Crossing_Analysis.ipynb
├── results/
│   ├── theta_star.pt
│   └── trajectories_*.parquet
├── launch_full_experiment.py  # Main launcher
├── gpu_monitor.py             # Real-time GPU monitoring
├── requirements.txt
└── README.md
```

## Technical Details

### Hypothesis
There exists a finite time t_c > 0 such that for two training trajectories
theta_h(t) (hot) and theta_c(t) (cold) with initial conditions:
  D_KL(theta_h(0) || theta^*) > D_KL(theta_c(0) || theta^*)

the following strict inequality holds:
  D_KL(theta_h(t_c) || theta^*) < D_KL(theta_c(t_c) || theta^*)

### Key Requirements from ТЗ
- **Batch size = 16**: Required for non-Gaussian heavy-tailed gradient noise
- **Tanh activation**: Required for Fisher information smoothness
- **Small batch -> heavy tails**: This is a NECESSARY condition for the effect
- **1000 independent runs**: Required for statistical significance

### Environment
| Component | Version |
|-----------|---------|
| Python | 3.11 |
| PyTorch | 2.12.0.dev+cu128 (nightly) |
| CUDA | 12.8 |
| GPU | RTX 5070 Ti Laptop (sm_120) |

## Honest Assessment

This codebase implements a **complete experimental pipeline** that:
- Correctly initializes "hot" and "cold" starting points
- Runs training trajectories on GPU
- Computes KL divergence and Fisher trace metrics
- Saves results for statistical analysis

What remains to be done:
1. **Pilot run** (50 trajectories) to detect if crossing exists at all
2. **Crossing detection analysis** if pilot shows promise
3. **Full 1000-run study** only after pilot confirms effect exists
4. **Peer review** of methodology and statistical rigor

Do NOT cite this as "verified Mpemba effect" — it is a **tool for investigating** the hypothesis.

## License
MIT
