"""
InfoMpemba: Full experiment launcher for RTX 5070 Ti (sm_120)
"""

import torch
import subprocess
import sys
import os
import time
import yaml
import argparse
from datetime import datetime


def check_gpu():
    """Check GPU availability and specs."""
    print("\n" + "=" * 70)
    print("GPU CHECK")
    print("=" * 70)

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available!")
        print("  Install: pip install --pre torch torchvision --index-url")
        print("    https://download.pytorch.org/whl/nightly/cu128")
        return None

    device_count = torch.cuda.device_count()
    print(f"OK: CUDA available: {device_count} GPU(s)")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA version: {torch.version.cuda}")

    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024 ** 3)
        print(f"\n  GPU {i}: {props.name}")
        print(f"    VRAM:    {vram_gb:.1f} GB")
        print(f"    Compute: {props.major}.{props.minor}")

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,power.max_limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            values = result.stdout.strip().split(", ")
            print(f"    Temp:  {values[0]} C")
            print(f"    Power: {values[1]} / {values[2]} W")
    except Exception:
        pass

    arch_list = torch.cuda.get_arch_list()
    if "sm_120" in arch_list:
        print(f"  sm_120 support: YES")
    else:
        print(f"  WARNING: sm_120 not in arch list!")

    return True


def set_gpu_power_limit(watts):
    """Set GPU power limit. Requires admin."""
    print(f"\nSetting GPU power limit: {watts} W")
    try:
        result = subprocess.run(
            ["nvidia-smi", "-pl", str(watts)],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            print(f"OK: Limit set to {watts}W")
            return True
        else:
            print(f"ERROR: {result.stderr.decode().strip()}")
            return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def print_recommendations():
    """Print launch recommendations."""
    print()
    print("=" * 70)
    print("LAUNCH OPTIONS")
    print("=" * 70)
    options = [
        "1. LF Pilot (LangevinFisher, batch=16, 30+30 runs, ~11h):",
        "   python launch_full_experiment.py --lf-pilot",
        "",
        "2. Full TZ-Compliant (LangevinFisher, batch=16, 1000 runs, ~15h):",
        "   python launch_full_experiment.py",
        "",
        "3. VanillaSGD Pilot (batch=16, 50+50 runs, ~2h):",
        "   python launch_full_experiment.py --pilot",
        "",
        "4. Quick Test (batch=16, 20 runs x 100 iter, ~10 min):",
        "   python launch_full_experiment.py --quick-test --no-confirm",
    ]
    print("\n".join(options))
    print()


def run_experiment(config_path, no_confirm=False):
    """Run experiment with given config."""
    if not os.path.exists(config_path):
        print(f"ERROR: Config not found: {config_path}")
        return False

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    print()
    print("=" * 70)
    print("EXPERIMENT LAUNCH")
    print("=" * 70)
    print(f"  Config:       {config_path}")
    print(f"  Experiment:   {config['experiment']['name']}")
    print(f"  Runs:         {config['experiment']['num_runs']} x 2 (hot + cold)")
    print(f"  Iterations:   {config['experiment']['max_iterations']}")
    print(f"  Batch size:   {config['data']['batch_size']}")
    print(f"  Optimizer:    {config['optimizer']['type']}")
    print(f"  Device:       {config['hardware']['device']}")

    num_runs = config['experiment']['num_runs']
    max_iter = config['experiment']['max_iterations']
    optimizer = config['optimizer']['type']

    # Time estimates based on measured benchmarks
    if optimizer == "LangevinFisher":
        time_per_iter = 0.085  # ~2x VanillaSGD (extra forward/backward for Fisher)
    else:
        time_per_iter = 0.047  # VanillaSGD baseline

    time_per_run = time_per_iter * max_iter
    total_time = num_runs * 2 * time_per_run

    print(f"\n  Time estimate:")
    print(f"     Per trajectory: ~{time_per_run:.1f} sec")
    print(f"     Total: ~{total_time/60:.0f} min ({total_time/3600:.1f} hours)")

    if not no_confirm:
        response = input(f"\n  Continue? (y/n): ").strip().lower()
        if response not in ['y', 'yes']:
            print("  Cancelled")
            return False
    else:
        print("\n  Auto-confirm (--no-confirm)")

    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n  Launch at {start_datetime}...")
    print(f"  {'='*70}\n")

    try:
        from src.mpemba_runner import run_experiment as run_mpemba
        df_results = run_mpemba(config)

        elapsed = time.time() - start_time
        end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print(f"\n{'='*70}")
        print("EXPERIMENT COMPLETE")
        print(f"{'='*70}")
        print(f"  Started:    {start_datetime}")
        print(f"  Finished:   {end_datetime}")
        print(f"  Elapsed:    {elapsed/3600:.2f} hours ({elapsed/60:.1f} min)")
        print(f"  Records:    {len(df_results)}")
        print(f"\n  Results: {config['output']['results_dir']}/{config['output']['parquet_file']}")
        print(f"  Analysis:  python analyze_lf_pilot.py")

        return True

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return False
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="InfoMpemba Launcher")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU and exit")
    parser.add_argument("--lf-pilot", action="store_true",
                        help="LangevinFisher pilot: 30+30 runs, 2000 steps, batch=16")
    parser.add_argument("--pilot", action="store_true",
                        help="VanillaSGD pilot: 50+50 runs, 500 steps, batch=16")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick test: 20 runs x 100 iter")
    parser.add_argument("--power-limit", type=int, default=None,
                        help="GPU power limit in watts (admin)")
    parser.add_argument("--config", type=str, default=None,
                        help="Custom config path")
    parser.add_argument("--no-confirm", action="store_true",
                        help="Skip confirmation prompt")

    args = parser.parse_args()

    print("=" * 70)
    print("InfoMpemba: RTX 5070 Ti Launcher")
    print("=" * 70)

    gpu_ok = check_gpu()

    if args.check_gpu:
        return

    if args.power_limit is not None:
        set_gpu_power_limit(args.power_limit)

    # Select config
    if args.config:
        config_path = args.config
    elif args.lf_pilot:
        config_path = "configs/lf_pilot.yaml"
    elif args.pilot:
        config_path = "configs/pilot.yaml"
    elif args.quick_test:
        config_path = "configs/quick_test.yaml"
        quick_config = {
            "experiment": {
                "name": "mpemba_quick_test",
                "num_runs": 20,
                "max_iterations": 100,
                "eval_interval": 25,
                "base_seed": 42
            },
            "model": {
                "layers": [784, 256, 128, 10],
                "activation": "tanh",
                "bias": True
            },
            "data": {
                "dataset": "FashionMNIST",
                "batch_size": 16,
                "val_samples_kl": 10000,
                "data_dir": "./data"
            },
            "optimizer": {
                "type": "VanillaSGD",
                "lr": 0.01,
                "momentum": 0.0,
                "noise_scale": 1.0
            },
            "initialization": {
                "cold_noise_std": 0.005,
                "hot_gain_multiplier": 3.0,
                "hot_laplace_scale": 0.5,
                "ref_epochs": 50,
                "ref_lr": 0.001,
                "ref_target_loss": 0.05
            },
            "hardware": {
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "num_workers": 0,
                "deterministic": False
            },
            "output": {
                "results_dir": "./results",
                "parquet_file": "trajectories_quick_test.parquet",
                "theta_star_file": "theta_star.pt",
                "log_interval": 5
            }
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(quick_config, f, default_flow_style=False, allow_unicode=True)
        print(f"\nOK: Quick test config created: {config_path}")
    else:
        config_path = "configs/full_tz.yaml"

    if not os.path.exists(config_path):
        print(f"\nERROR: Config not found: {config_path}")
        return

    print_recommendations()

    success = run_experiment(config_path, no_confirm=args.no_confirm)

    if success:
        print("\nDone! Analysis:")
        print("   python analyze_lf_pilot.py")


if __name__ == "__main__":
    main()
