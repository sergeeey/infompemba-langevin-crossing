"""Cold-only drift test: is the cold trajectory stationary near theta_star?"""

import torch
import yaml
import os
import sys
import copy
import pandas as pd
import numpy as np
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.initializers import create_model, initialize_all
from src.langevin_sgd import LangevinSGD
from src.fisher_metrics import create_metric_analyzer
from src.mpemba_runner import create_dataloaders, setup_deterministic, run_single_trajectory


def main():
    with open("configs/lf_pilot.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["experiment"]["num_runs"] = 10
    config["experiment"]["max_iterations"] = 30000
    config["experiment"]["eval_interval"] = 50

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader = create_dataloaders(config)

    base_seed = config["experiment"]["base_seed"]
    setup_deterministic(base_seed, config["hardware"]["deterministic"])
    model_star, model_cold, _ = initialize_all(config, device, seed=base_seed)

    analyzer = create_metric_analyzer(
        model_star=model_star, val_loader=val_loader, config=config, device=device
    )

    parquet_path = "results/trajectories_cold_drift_test.parquet"
    all_results = []
    cooldown = config["hardware"].get("cooldown_seconds", 0)

    completed = set()
    if os.path.exists(parquet_path):
        df_existing = pd.read_parquet(parquet_path)
        for _, grp in df_existing.groupby("run_id"):
            completed.add(int(grp["run_id"].iloc[0]))
        all_results = df_existing.to_dict("records")
        print(f"Resuming: {len(completed)} runs already done")

    init_weights = model_cold.state_dict()
    num_runs = config["experiment"]["num_runs"]

    print(f"\n=> Cold-only drift test: {num_runs} runs x {config['experiment']['max_iterations']} iter")
    start = time.time()

    for run_idx in range(num_runs):
        if run_idx in completed:
            continue

        run_seed = base_seed + run_idx
        setup_deterministic(run_seed, config["hardware"]["deterministic"])

        results = run_single_trajectory(
            init_weights=init_weights,
            train_loader=train_loader,
            config=config,
            analyzer=analyzer,
            device=device,
            run_id=run_idx,
            init_type="cold",
        )
        all_results.extend(results)

        pd.DataFrame(all_results).to_parquet(parquet_path, index=False)
        torch.cuda.empty_cache()

        if cooldown > 0:
            time.sleep(cooldown)

        print(f"   [{run_idx+1}/{num_runs}] done")

    elapsed = time.time() - start
    print(f"\n=> Done in {elapsed/60:.1f} min")

    df = pd.DataFrame(all_results)
    means = df.groupby("step")["kl_div_star"].agg(["mean", "std"])
    kl_start = means["mean"].iloc[0]
    kl_end = means["mean"].iloc[-1]
    drift_pct = (kl_end - kl_start) / kl_start * 100

    print(f"\n{'='*50}")
    print(f"COLD DRIFT TEST RESULT")
    print(f"{'='*50}")
    print(f"KL start: {kl_start:.4f}")
    print(f"KL end:   {kl_end:.4f}")
    print(f"Drift:    {drift_pct:+.1f}%")
    print(f"Std end:  {means['std'].iloc[-1]:.4f}")

    if abs(drift_pct) > 20:
        print(f"\n>>> FAIL: cold drifts {drift_pct:+.1f}% -- NOT stationary")
        print(f">>> Experiment is INVALID without fixing cold stability")
    else:
        print(f"\n>>> PASS: cold drift {drift_pct:+.1f}% -- acceptable (<20%)")


if __name__ == "__main__":
    main()
