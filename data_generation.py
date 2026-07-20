"""
data_generation.py -- all SYNTHETIC data generation for the GPU cluster
scheduler.

Contains:
  - generate_gpu_price_curve()  -> data/gpu_prices.csv
  - generate_jobs()             -> ordinary random synthetic jobs
  - generate_edge_case_jobs()   -> guaranteed tight-deadline/high-priority
                                    jobs (forced regardless of seed, so the
                                    demo reliably shows the reward
                                    function's penalty terms mattering)

CLI:
    python data_generation.py prices      # -> data/gpu_prices.csv
    python data_generation.py jobs        # -> data/jobs.csv (100% synthetic)
    python data_generation.py all         # both (default if no arg given)
"""

import sys

import numpy as np
import pandas as pd

SEED = 42

# --- gpu_prices.csv ---
LOW_NIGHT = 0.38       # $/GPU-hour, overnight trough
MID_MORNING = 0.75
HIGH_AFTERNOON = 1.10
PEAK_EVENING = 1.60
PRICE_NOISE_STD = 0.04
MIN_PRICE = 0.05
MAX_PRICE = 2.00

# --- jobs.csv (random + edge-case) ---
N_JOBS = 200
PRIORITIES = ["Low", "Medium", "High"]
PRIORITY_WEIGHTS = [0.5, 0.35, 0.15]  # most jobs are low/medium priority
MAX_GPU_CHOICES = [2, 4, 6, 8]
MAX_GPU_WEIGHTS = [0.35, 0.35, 0.20, 0.10]


def generate_gpu_price_curve(seed: int = SEED) -> pd.DataFrame:
    """Synthetic hourly GPU spot-price curve (no free real historical
    GPU-price series exists to substitute here)."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24)

    anchor_hours = np.array([0, 8, 14, 18, 23])
    anchor_prices = np.array(
        [LOW_NIGHT, MID_MORNING, HIGH_AFTERNOON, PEAK_EVENING, LOW_NIGHT + 0.15]
    )
    base = np.interp(hours, anchor_hours, anchor_prices)

    noise = rng.normal(0, PRICE_NOISE_STD, size=24)
    price = np.clip(base + noise, MIN_PRICE, MAX_PRICE)

    return pd.DataFrame({"hour": hours, "gpu_price": np.round(price, 2)})


def generate_jobs(n_jobs: int = N_JOBS, seed: int = SEED) -> pd.DataFrame:
    """Ordinary random synthetic AI workload."""
    rng = np.random.default_rng(seed)

    arrival_time = np.round(rng.uniform(0, 22, size=n_jobs), 1)
    priority = rng.choice(PRIORITIES, size=n_jobs, p=PRIORITY_WEIGHTS)
    max_gpus = rng.choice(MAX_GPU_CHOICES, size=n_jobs, p=MAX_GPU_WEIGHTS)

    base_hours = rng.lognormal(mean=1.6, sigma=0.9, size=n_jobs)
    gpu_hours_required = np.clip(base_hours * (max_gpus / 4), 1, 96)
    gpu_hours_required = np.round(gpu_hours_required, 1)

    min_hours_needed = gpu_hours_required / max_gpus
    slack_factor = rng.uniform(1.1, 4.0, size=n_jobs)
    deadline = arrival_time + np.maximum(min_hours_needed * slack_factor, 1.0)
    deadline = np.clip(deadline, arrival_time + 1, 24.0)
    deadline = np.round(deadline, 1)

    return pd.DataFrame(
        {
            "job_id": [f"synthetic_{i}" for i in range(n_jobs)],
            "arrival_time": arrival_time,
            "deadline": deadline,
            "gpu_hours_required": gpu_hours_required,
            "priority": priority,
            "max_gpus": max_gpus,
            "initial_progress": np.zeros(n_jobs),
        }
    ).sort_values("arrival_time").reset_index(drop=True)


def generate_edge_case_jobs(n_jobs: int, seed: int = SEED) -> pd.DataFrame:
    """Guaranteed tight-deadline / high-priority jobs, forced regardless of
    seed (unlike generate_jobs()'s random slack_factor, which only
    sometimes produces a near-infeasible job by chance). Used by
    real_data.py to top up the real Alibaba-sourced rows."""
    rng = np.random.default_rng(seed + 1)  # different stream than generate_jobs()

    arrival_time = np.round(rng.uniform(0, 18, size=n_jobs), 1)
    priority = rng.choice(["High", "Medium"], size=n_jobs, p=[0.7, 0.3])
    max_gpus = rng.choice([2, 4], size=n_jobs, p=[0.6, 0.4])

    base_hours = rng.lognormal(mean=2.0, sigma=0.6, size=n_jobs)
    gpu_hours_required = np.clip(base_hours * (max_gpus / 4), 4, 96)
    gpu_hours_required = np.round(gpu_hours_required, 1)

    min_hours_needed = gpu_hours_required / max_gpus
    slack_factor = rng.uniform(1.0, 1.3, size=n_jobs)  # forced near/at infeasible
    deadline = arrival_time + np.maximum(min_hours_needed * slack_factor, 1.0)
    deadline = np.clip(deadline, arrival_time + 1, 24.0)
    deadline = np.round(deadline, 1)

    return pd.DataFrame(
        {
            "job_id": [f"edge_{i}" for i in range(n_jobs)],
            "arrival_time": arrival_time,
            "deadline": deadline,
            "gpu_hours_required": gpu_hours_required,
            "priority": priority,
            "max_gpus": max_gpus,
            "initial_progress": np.zeros(n_jobs),
        }
    )


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode in ("prices", "all"):
        prices_df = generate_gpu_price_curve()
        prices_df.to_csv("data/gpu_prices.csv", index=False)
        print(prices_df.to_string(index=False))
        print("Saved -> data/gpu_prices.csv\n")

    if mode in ("jobs", "all"):
        jobs_df = generate_jobs()
        jobs_df.to_csv("data/jobs.csv", index=False)
        print(jobs_df.head(10).to_string(index=False))
        print(f"\n{len(jobs_df)} jobs generated -> data/jobs.csv (100% synthetic -- "
              f"run real_data.py instead for the real+synthetic mix)")
