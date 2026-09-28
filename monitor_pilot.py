"""
Pilot run progress monitor.
Run in separate window while pilot is executing.
"""

import os
import time
import pandas as pd
from datetime import datetime


def monitor_progress():
    """Monitor pilot progress in real-time."""
    parquet_path = "results/trajectories_pilot.parquet"
    
    print("=" * 70)
    print("InfoMpemba Pilot Monitor")
    print("=" * 70)
    print()
    print("Waiting for pilot to start producing results...")
    print("(Press Ctrl+C to stop)")
    print()
    
    last_count = 0
    last_kl_cold = None
    last_kl_hot = None
    
    try:
        while True:
            if os.path.exists(parquet_path):
                try:
                    df = pd.read_parquet(parquet_path)
                    
                    total_runs = df['run_id'].nunique() if len(df) > 0 else 0
                    cold_runs = df[df['init_type'] == 'cold']['run_id'].nunique() if len(df) > 0 else 0
                    hot_runs = df[df['init_type'] == 'hot']['run_id'].nunique() if len(df) > 0 else 0
                    total_records = len(df)
                    
                    # Current KL stats
                    last_cold = df[df['init_type'] == 'cold'].groupby('run_id').last()
                    last_hot = df[df['init_type'] == 'hot'].groupby('run_id').last()
                    
                    avg_kl_cold = last_cold['kl_div_star'].mean() if len(last_cold) > 0 else None
                    avg_kl_hot = last_hot['kl_div_star'].mean() if len(last_hot) > 0 else None
                    
                    now = datetime.now().strftime("%H:%M:%S")
                    
                    # Clear line
                    print(f"\r{' ' * 80}", end="")
                    print(f"\r{now} | Runs: {cold_runs+hot_runs}/100 (cold={cold_runs}, hot={hot_runs}) | Records: {total_records} | Avg KL cold={avg_kl_cold:.4f} hot={avg_kl_hot:.4f}" if avg_kl_cold is not None else f"\r{now} | Runs: {cold_runs+hot_runs}/100 | Records: {total_records}", end="", flush=True)
                    
                    # Check for crossing
                    if len(last_cold) > 5 and len(last_hot) > 5:
                        if avg_kl_hot < avg_kl_cold:
                            print(f" *** POSSIBLE CROSSING! hot < cold ***", end="", flush=True)
                    
                    last_count = total_runs
                    last_kl_cold = avg_kl_cold
                    last_kl_hot = avg_kl_hot
                    
                except Exception:
                    # File might be being written
                    pass
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        # Final summary
        if os.path.exists(parquet_path):
            try:
                df = pd.read_parquet(parquet_path)
                print(f"\nFinal results: {len(df)} records, {df['run_id'].nunique()} runs")
                for init_type in ['cold', 'hot']:
                    subset = df[df['init_type'] == init_type]
                    if len(subset) > 0:
                        last = subset.groupby('run_id').last()
                        print(f"  {init_type}: mean KL = {last['kl_div_star'].mean():.4f}, std = {last['kl_div_star'].std():.4f}")
            except Exception:
                pass


if __name__ == "__main__":
    monitor_progress()
