"""
evaluate.py -- runs the trained agent job-by-job through
GPUClusterEnv(test=True) (one episode per job, deterministic order) --
the per-episode unit GPUClusterEnv already works in, not baseline.py's
whole-queue shared-pool simulation. Those are structurally different and
not comparable at the per-step level; what is comparable is the aggregate:
how many jobs did the agent finish, how many missed deadline, at what
cost -- lined up against baseline.py's FCFS/Always-Max/Priority table.

Usage:
    python evaluate.py [checkpoint_path]
    (defaults to config.CHECKPOINT_DIR/dqn_final.pt)
"""
import sys

import numpy as np
import pandas as pd
import torch

import config
import data_preprocessing as dp
from environment import GPUClusterEnv, ACTIONS
from model import DuelingMLP
from baseline import run_all_baselines, compute_hourly_demand, PRIORITY_RANK


def evaluate_agent(model, jobs, prices, demand_curve=None, device="cpu"):
    """
    One episode per job (deterministic test-mode order). Greedy policy
    (argmax, no exploration) -- this is evaluation, not training.

    demand_curve, if given, is used with enforce_capacity=False: the
    model sees a real cluster_utilization reading (matching what it was
    trained on) but this job's own allocation is not capacity-limited --
    that's what makes this "isolated": what this job's outcome would be
    if it alone had priority, while still showing the model realistic
    inputs rather than the fallback synthetic curve it was never trained
    on.

    Returns a dict shaped like baseline.py's rows, plus total_reward, so
    the two are directly comparable in one table.
    """
    model.eval()
    env = GPUClusterEnv(jobs=jobs, prices=prices, demand_curve=demand_curve,
                         enforce_capacity=False, test=True)

    completed = 0
    missed = 0
    total_cost = 0.0
    total_reward = 0.0
    per_job_records = []

    for _ in range(len(jobs)):
        state = env.reset()
        job_id = env.job["job_id"]
        done = False
        ep_reward = 0.0
        ep_cost = 0.0

        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
            with torch.no_grad():
                # Raw action index in, straight to step() -- GPUClusterEnv
                # resolves it internally.
                action = model(state_t).argmax(dim=1).item()
            state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_cost += info["cost"]

        job_complete = info["job_progress"] >= 1.0
        if job_complete:
            completed += 1
        if info["deadline_missed"]:
            missed += 1

        total_cost += ep_cost
        total_reward += ep_reward
        per_job_records.append({
            "job_id": job_id,
            "completed": job_complete,
            "deadline_missed": info["deadline_missed"],
            "cost": round(ep_cost, 2),
            "reward": round(ep_reward, 2),
        })

    model.train()
    n_jobs = len(jobs)
    return {
        "policy": "rl_agent",
        "total_cost": round(total_cost, 2),
        "jobs_completed": completed,
        "deadlines_missed": missed,
        "deadline_success_rate_pct": round(completed / n_jobs * 100.0, 1) if n_jobs else 0.0,
        "total_reward": round(total_reward, 2),
    }, per_job_records


def _build_state(hour, price, job_progress, deadline_remaining, gpu_hours_remaining, cluster_utilization, priority):
    h = hour % 24
    hour_sin = np.sin(2 * np.pi * h / 24)
    hour_cos = np.cos(2 * np.pi * h / 24)
    urgency_ratio = float(
        np.clip(gpu_hours_remaining / max(deadline_remaining, 0.5), 0.0, 5.0)
    )
    priority_rank = config.PRIORITY_RANK_VALUE[priority]
    return np.array(
        [hour_sin, hour_cos, price, job_progress, deadline_remaining,
         gpu_hours_remaining, cluster_utilization, urgency_ratio, priority_rank],
        dtype=np.float32,
    )


def evaluate_agent_shared_pool(model, jobs, prices, device="cpu", total_gpus=None, episode_length=None):
    """
    The number directly comparable to baseline.py -- the same hour-by-hour,
    shared-capacity engine, except each active job's allocation this hour
    is whatever the trained agent's greedy policy requests for that job's
    own state, instead of a fixed heuristic.

    Caveat that belongs in the README, not just here: the agent never saw
    multi-job contention during training (GPUClusterEnv is one-job-per-
    episode by design). When total requested GPUs exceed the pool, this
    function arbitrates by priority then deadline -- that rule is not
    something the agent learned, it's a necessary bolt-on to run a
    single-job-trained policy in a multi-job setting. True joint /
    multi-agent scheduling (the agent itself learning to yield GPUs to a
    more urgent job) is out of scope.
    """
    model.eval()
    total_gpus = total_gpus or config.TOTAL_CLUSTER_GPUS
    episode_length = episode_length or config.EPISODE_LENGTH

    state = {
        j["job_id"]: {
            "remaining": float(j["gpu_hours_required"]),
            "required": float(j["gpu_hours_required"]),
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
        if not eligible_ids:
            continue

        # Ask the agent what each eligible job wants this hour (batched
        # through the network in one forward pass). cluster_utilization
        # here is computed live from the exact jobs competing this hour --
        # more precise than train.py's precomputed worst-case curve, since
        # this function already has to enumerate eligible_ids anyway.
        total_eligible_demand = sum(state[jid]["max_gpus"] for jid in eligible_ids)

        state_vecs = []
        for jid in eligible_ids:
            s = state[jid]
            job_progress = 1.0 - (s["remaining"] / s["required"]) if s["required"] > 0 else 1.0
            deadline_remaining = max(0.0, s["deadline"] - hour)
            background_demand = max(0.0, total_eligible_demand - s["max_gpus"])
            cluster_util = float(np.clip(background_demand / total_gpus, 0.0, 1.0))
            state_vecs.append(_build_state(
                hour, price, job_progress, deadline_remaining, s["remaining"], cluster_util, s["priority"]
            ))

        batch = torch.from_numpy(np.stack(state_vecs)).float().to(device)
        with torch.no_grad():
            action_idxs = model(batch).argmax(dim=1).tolist()

        requests = {}
        for jid, a in zip(eligible_ids, action_idxs):
            s = state[jid]
            requests[jid] = min(ACTIONS[a], s["max_gpus"], s["remaining"])

        # Capacity arbitration -- priority first, deadline as tiebreak
        # within a priority tier (matches baseline.py's "priority" policy
        # ordering).
        order = sorted(eligible_ids, key=lambda jid: (PRIORITY_RANK[state[jid]["priority"]], state[jid]["deadline"]))

        gpus_left = total_gpus
        for jid in order:
            if gpus_left <= 0:
                break
            give = min(requests[jid], gpus_left)
            if give <= 0:
                continue
            s = state[jid]
            s["remaining"] -= give
            gpus_left -= give
            total_cost += price * give
            total_gpu_hours_used += give
            if s["remaining"] <= 1e-9:
                s["completed"] = True

    model.train()
    completed = sum(1 for s in state.values() if s["completed"])
    missed = sum(1 for s in state.values() if s["missed"] and not s["completed"])
    n_jobs = len(state)
    utilization = total_gpu_hours_used / (total_gpus * episode_length) * 100.0

    return {
        "policy": "rl_agent_shared_pool",
        "gpu_utilization_pct": round(utilization, 1),
        "total_cost": round(total_cost, 2),
        "jobs_completed": completed,
        "deadlines_missed": missed,
        "deadline_success_rate_pct": round(completed / n_jobs * 100.0, 1) if n_jobs else 0.0,
    }


def main(checkpoint_path=None):
    checkpoint_path = checkpoint_path or f"{config.CHECKPOINT_DIR}/dqn_final.pt"

    print(f"Loading checkpoint: {checkpoint_path}")
    model = DuelingMLP(config.STATE_DIM, config.ACTION_DIM)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    print("Loading data...")
    jobs = dp.load_job_data()
    prices = dp.load_price_data()

    print("\nRunning baseline comparison (shared 32-GPU pool, all jobs competing)...")
    baseline_df = run_all_baselines(jobs=jobs, prices=prices)

    print("Running agent through the SAME shared-pool simulation "
          "(fair comparison)...")
    shared_row = evaluate_agent_shared_pool(model, jobs, prices)

    print("\n" + "=" * 78)
    print("  HEADLINE COMPARISON -- same shared 32-GPU pool for everyone")
    print("=" * 78)
    full_df = pd.concat([baseline_df, pd.DataFrame([shared_row])], ignore_index=True)
    print(full_df.to_string(index=False))

    best_baseline = baseline_df.loc[baseline_df["deadline_success_rate_pct"].idxmax()]
    delta = shared_row["deadline_success_rate_pct"] - best_baseline["deadline_success_rate_pct"]
    print(f"\n  Agent vs. best baseline ({best_baseline['policy']}): "
          f"{delta:+.1f} points deadline success rate, at "
          f"${shared_row['total_cost'] - best_baseline['total_cost']:+.2f} cost delta")
    print("=" * 78)

    print("\nRunning isolated per-job evaluation (diagnostic only -- not "
          "comparable to the table above, no shared-capacity constraint, "
          "exclude from the pitch)...")
    full_demand = compute_hourly_demand(jobs)
    isolated_row, per_job = evaluate_agent(model, jobs, prices, demand_curve=full_demand)
    print(f"  isolated: cost={isolated_row['total_cost']}, "
          f"completed={isolated_row['jobs_completed']}, "
          f"success_rate={isolated_row['deadline_success_rate_pct']}% "
          f"-- inflated vs. the headline number above, ignore for reporting")

    return shared_row, baseline_df, isolated_row


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    main(ckpt)
