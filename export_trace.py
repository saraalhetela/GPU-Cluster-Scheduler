"""
export_trace.py -- records a REAL hour-by-hour trace of the shared-pool
simulation, for the dashboard demo.

This does NOT reimplement new simulation logic -- it mirrors
baseline.py's simulate_policy("priority") and evaluate.py's
evaluate_agent_shared_pool() exactly (same eligibility rule, same
arbitration order, same everything), it just additionally records a
per-hour snapshot instead of only returning the final aggregate. The
dashboard animates this trace instead of jumping straight to endpoint
numbers.

Usage:
    python export_trace.py [checkpoint_path]
    -> trace.json
"""
import json
import sys

import numpy as np
import torch

import config
import data_preprocessing as dp
from environment import ACTIONS
from model import DuelingMLP

PRIORITY_RANK = {"High": 0, "Medium": 1, "Low": 2}


def _order_priority(jobs):
    return sorted(jobs, key=lambda j: (PRIORITY_RANK[j["priority"]], j["arrival_time"]))


def trace_priority(jobs, prices, total_gpus=None, episode_length=None):
    """Mirrors baseline.simulate_policy(jobs, prices, "priority")."""
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

    cum_cost = 0.0
    trace = []

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
        ordered = _order_priority(eligible_jobs)

        gpus_left = total_gpus
        gpus_used_this_hour = 0.0
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
            cum_cost += price * give
            gpus_used_this_hour += give
            if s["remaining"] <= 1e-9:
                s["completed"] = True

        trace.append({
            "hour": hour,
            "price": round(price, 3),
            "active_jobs": len(eligible_ids),
            "gpus_used": round(gpus_used_this_hour, 1),
            "cum_cost": round(cum_cost, 2),
            "cum_completed": sum(1 for s in state.values() if s["completed"]),
            "cum_missed": sum(1 for s in state.values() if s["missed"] and not s["completed"]),
        })

    return trace


def _build_state_vec(hour, price, job_progress, deadline_remaining, gpu_hours_remaining, cluster_utilization, priority):
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


def trace_agent_shared_pool(model, jobs, prices, device="cpu", total_gpus=None, episode_length=None):
    """Mirrors evaluate.evaluate_agent_shared_pool() exactly."""
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
            "completed": False,
            "missed": False,
        }
        for j in jobs
    }

    cum_cost = 0.0
    trace = []

    for hour in range(episode_length):
        price = prices.get(hour % 24, sum(prices.values()) / len(prices))

        for s in state.values():
            if not s["completed"] and not s["missed"] and hour >= s["deadline"]:
                s["missed"] = True

        eligible_ids = [
            jid for jid, s in state.items()
            if s["arrival_time"] <= hour < s["deadline"] and not s["completed"] and not s["missed"]
        ]

        gpus_used_this_hour = 0.0

        if eligible_ids:
            total_eligible_demand = sum(state[jid]["max_gpus"] for jid in eligible_ids)
            state_vecs = []
            for jid in eligible_ids:
                s = state[jid]
                job_progress = 1.0 - (s["remaining"] / s["required"]) if s["required"] > 0 else 1.0
                deadline_remaining = max(0.0, s["deadline"] - hour)
                background_demand = max(0.0, total_eligible_demand - s["max_gpus"])
                cluster_util = float(np.clip(background_demand / total_gpus, 0.0, 1.0))
                state_vecs.append(_build_state_vec(
                    hour, price, job_progress, deadline_remaining, s["remaining"], cluster_util
                ))

            batch = torch.from_numpy(np.stack(state_vecs)).float().to(device)
            with torch.no_grad():
                action_idxs = model(batch).argmax(dim=1).tolist()

            requests = {}
            for jid, a in zip(eligible_ids, action_idxs):
                s = state[jid]
                requests[jid] = min(ACTIONS[a], s["max_gpus"], s["remaining"])

            order = sorted(eligible_ids, key=lambda jid: state[jid]["deadline"])
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
                cum_cost += price * give
                gpus_used_this_hour += give
                if s["remaining"] <= 1e-9:
                    s["completed"] = True

        trace.append({
            "hour": hour,
            "price": round(price, 3),
            "active_jobs": len(eligible_ids),
            "gpus_used": round(gpus_used_this_hour, 1),
            "cum_cost": round(cum_cost, 2),
            "cum_completed": sum(1 for s in state.values() if s["completed"]),
            "cum_missed": sum(1 for s in state.values() if s["missed"] and not s["completed"]),
        })

    model.train()
    return trace


def main(checkpoint_path=None):
    checkpoint_path = checkpoint_path or f"{config.CHECKPOINT_DIR}/dqn_final.pt"

    print(f"Loading checkpoint: {checkpoint_path}")
    model = DuelingMLP(config.STATE_DIM, config.ACTION_DIM)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    print("Loading data...")
    jobs = dp.load_job_data()
    prices = dp.load_price_data()

    print("Tracing priority baseline (best baseline, for the race)...")
    priority_trace = trace_priority(jobs, prices)

    print("Tracing RL agent (shared pool)...")
    agent_trace = trace_agent_shared_pool(model, jobs, prices)

    out = {
        "n_jobs": len(jobs),
        "total_gpus": config.TOTAL_CLUSTER_GPUS,
        "episode_length": config.EPISODE_LENGTH,
        "priority": priority_trace,
        "rl_agent_shared_pool": agent_trace,
    }

    with open("trace.json", "w") as f:
        json.dump(out, f, indent=2)

    print("\nSaved -> trace.json")
    print(f"  priority final:  cost=${priority_trace[-1]['cum_cost']:.2f}  "
          f"completed={priority_trace[-1]['cum_completed']}  missed={priority_trace[-1]['cum_missed']}")
    print(f"  rl_agent final:  cost=${agent_trace[-1]['cum_cost']:.2f}  "
          f"completed={agent_trace[-1]['cum_completed']}  missed={agent_trace[-1]['cum_missed']}")
    print("\nPaste the contents of trace.json back to Claude to build the animated demo.")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else None
    main(ckpt)
