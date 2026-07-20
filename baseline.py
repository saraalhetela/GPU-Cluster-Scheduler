"""
baseline.py -- whole-queue heuristic comparison for the GPU cluster
scheduler pitch.

Policies:
    FCFS        -- earliest arrival_time served first
    Always Max  -- arbitrary (jobs.csv) order, every active job gets its
                   max_gpus if the pool has room
    Priority    -- High > Medium > Low served first, arrival_time as
                   tiebreak within a priority tier

Usage:
    python baseline.py
"""
import numpy as np
import pandas as pd

import config
import data_preprocessing as dp

PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}  # lower rank = served first


def compute_hourly_demand(jobs, episode_length=None):
    episode_length = episode_length or config.EPISODE_LENGTH
    demand = np.zeros(episode_length, dtype=np.float64)
    for j in jobs:
        arrival = float(j["arrival_time"])
        deadline = float(j["deadline"])
        duration = max(deadline - arrival, 0.01)
        # average concurrent GPU need, not peak -- max_gpus is a ceiling the
        # job is *allowed* to request, not what it needs for its full window
        avg_gpus = float(j["gpu_hours_required"]) / duration
        start_hour = max(0, int(np.floor(arrival)))
        end_hour = min(episode_length, int(np.ceil(deadline)))
        for h in range(start_hour, end_hour):
            demand[h] += avg_gpus
    return demand


def _order_jobs(jobs, policy):
    if policy == "fcfs":
        return sorted(jobs, key=lambda j: j["arrival_time"])
    elif policy == "always_max":
        return list(jobs)  # jobs.csv's own order, arbitrary on purpose
    elif policy == "priority":
        return sorted(jobs, key=lambda j: (PRIORITY_RANK[j["priority"]], j["arrival_time"]))
    raise ValueError(f"unknown policy: {policy}")


def simulate_policy(jobs, prices, policy, total_gpus=None, episode_length=None):
    total_gpus = total_gpus or config.TOTAL_CLUSTER_GPUS
    episode_length = episode_length or config.EPISODE_LENGTH

    state = {
        j["job_id"]: {
            "remaining": float(j["gpu_hours_required"]),
            "max_gpus": float(j["max_gpus"]),
            "arrival_time": float(j["arrival_time"]),
            "deadline": float(j["deadline"]),
            "priority": j["priority"],
            "completed": False,
            "missed": False,
        }
        for j in jobs
    }

    total_cost = 0.0
    total_gpu_hours_used = 0.0

    for hour in range(episode_length):
        price = prices.get(hour % 24, sum(prices.values()) / len(prices))

        for s in state.values():
            if not s["completed"] and not s["missed"] and hour >= s["deadline"]:
                s["missed"] = True

        eligible_ids = [
            jid for jid, s in state.items()
            if s["arrival_time"] <= hour < s["deadline"] and not s["completed"] and not s["missed"]
        ]
        eligible_jobs = [j for j in jobs if j["job_id"] in eligible_ids]
        ordered = _order_jobs(eligible_jobs, policy)

        gpus_left = total_gpus
        for j in ordered:
            if gpus_left <= 0:
                break
            s = state[j["job_id"]]
            want = min(s["max_gpus"], s["remaining"])
            give = min(want, gpus_left)
            if give <= 0:
                continue

            s["remaining"] -= give
            gpus_left -= give
            total_cost += price * give
            total_gpu_hours_used += give

            if s["remaining"] <= 1e-9:
                s["completed"] = True

    completed = sum(1 for s in state.values() if s["completed"])
    missed = sum(1 for s in state.values() if s["missed"] and not s["completed"])
    n_jobs = len(state)
    utilization = total_gpu_hours_used / (total_gpus * episode_length) * 100.0

    return {
        "policy": policy,
        "gpu_utilization_pct": round(utilization, 1),
        "total_cost": round(total_cost, 2),
        "jobs_completed": completed,
        "deadlines_missed": missed,
        "deadline_success_rate_pct": round(completed / n_jobs * 100.0, 1) if n_jobs else 0.0,
    }


def run_all_baselines(jobs=None, prices=None):
    jobs = jobs if jobs is not None else dp.load_job_data()
    prices = prices if prices is not None else dp.load_price_data()

    rows = [simulate_policy(jobs, prices, p) for p in ("fcfs", "always_max", "priority")]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_all_baselines()
    print(df.to_string(index=False))
