"""
test_env.py -- sanity tests for GPUClusterEnv: random-policy rollouts +
invariant checks, run before wiring the env into train.py so bugs get
caught here instead of showing up as a training run that silently never
learns.

Usage:
    python test_env.py
Exits non-zero if any check fails, so it can be dropped into CI/pre-commit
too, not just run by hand.
"""

import sys

import numpy as np

import config
import data_preprocessing as dp
from environment import GPUClusterEnv
from baseline import compute_hourly_demand

N_EPISODES = 50
SEED = 0


def run_random_policy_episode(env, rng) -> dict:
    state = env.reset()
    assert state.shape == (config.STATE_DIM,), f"bad state shape {state.shape}"

    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = rng.integers(0, config.ACTION_DIM)
        next_state, reward, done, info = env.step(action)

        assert next_state.shape == (config.STATE_DIM,), "state shape changed mid-episode"
        assert not np.isnan(next_state).any(), "NaN in state"
        assert not np.isinf(next_state).any(), "inf in state"
        assert not (np.isnan(reward) or np.isinf(reward)), "bad reward"
        assert 0.0 <= next_state[3] <= 1.0, f"job_progress out of bounds: {next_state[3]}"
        assert next_state[4] >= 0.0, f"deadline_remaining went negative: {next_state[4]}"
        assert next_state[5] >= -1e-6, f"gpu_hours_remaining went negative: {next_state[5]}"
        assert 0.0 <= next_state[6] <= 1.0, f"cluster_utilization out of bounds: {next_state[6]}"

        total_reward += reward
        steps += 1
        state = next_state

        assert steps <= config.EPISODE_LENGTH, "episode ran past EPISODE_LENGTH without done=True"

    return {"steps": steps, "total_reward": total_reward, "info": info}


def test_random_policy_many_episodes():
    prices = dp.load_price_data()
    jobs = dp.load_job_data()
    demand = compute_hourly_demand(jobs)
    env = GPUClusterEnv(jobs=jobs, prices=prices, demand_curve=demand)
    rng = np.random.default_rng(SEED)

    results = [run_random_policy_episode(env, rng) for _ in range(N_EPISODES)]
    completed = sum(1 for r in results if r["info"]["job_progress"] >= 1.0)
    missed = sum(1 for r in results if r["info"]["deadline_missed"])
    print(f"[random policy, real demand curve] {N_EPISODES} episodes: "
          f"{completed} completed, {missed} deadline-missed, "
          f"mean reward {np.mean([r['total_reward'] for r in results]):.2f}")


def test_capacity_enforcement_limits_allocation():
    prices = dp.load_price_data()
    jobs = dp.load_job_data()

    saturated_curve = np.full(config.EPISODE_LENGTH, 10_000.0)
    env = GPUClusterEnv(jobs=jobs, prices=prices, demand_curve=saturated_curve, test=True)

    env.reset()
    max_action = config.ACTION_DIM - 1
    _, reward, done, info = env.step(max_action)

    assert info["gpus_allocated"] <= 0.5, (
        f"expected near-zero allocation under full saturation, got "
        f"{info['gpus_allocated']} (requested {info['requested_gpus']})"
    )
    print(f"[capacity enforcement] fully-saturated hour -> "
          f"gpus_allocated={info['gpus_allocated']} (requested {info['requested_gpus']}, expected ~0)")

    empty_curve = np.zeros(config.EPISODE_LENGTH)
    env2 = GPUClusterEnv(jobs=jobs, prices=prices, demand_curve=empty_curve, test=True)
    env2.reset()
    _, reward2, done2, info2 = env2.step(max_action)
    expected = min(config.GPU_ACTIONS[-1], env2.max_gpus)
    assert info2["gpus_allocated"] == expected, (
        f"expected full allocation ({expected}) with zero contention, got {info2['gpus_allocated']}"
    )
    print(f"[capacity enforcement] zero-contention hour -> "
          f"gpus_allocated={info2['gpus_allocated']} (expected {expected}, unconstrained)")


def test_zero_allocation_always_penalized_or_neutral():
    prices = dp.load_price_data()
    jobs = dp.load_job_data()
    env = GPUClusterEnv(jobs=jobs, prices=prices, test=True)
    
    env._test_idx = 0
    env.reset()
    zero_reward = 0.0
    done = False
    while not done:
        _, r, done, _ = env.step(0)  # action 0 == 0 GPUs
        zero_reward += r

    env._test_idx = 0
    env.reset()
    max_action = config.ACTION_DIM - 1
    max_reward = 0.0
    done = False
    while not done:
        _, r, done, info = env.step(max_action)
        max_reward += r

    assert max_reward >= zero_reward, (
        f"allocating max GPUs scored worse ({max_reward:.2f}) than allocating "
        f"zero ({zero_reward:.2f}) on the same job -- check reward calibration"
    )
    print(f"[reward sanity] max-alloc reward {max_reward:.2f} >= zero-alloc reward {zero_reward:.2f}")


def test_deterministic_reset_in_test_mode():
    prices = dp.load_price_data()
    jobs = dp.load_job_data()
    env = GPUClusterEnv(jobs=jobs, prices=prices, test=True)

    seen_job_ids = []
    for _ in range(min(5, len(jobs))):
        env.reset()
        seen_job_ids.append(env.job["job_id"])

    expected = [j["job_id"] for j in jobs[:5]]
    assert seen_job_ids == expected, f"test-mode reset order wrong: {seen_job_ids} != {expected}"
    print(f"[determinism] test-mode job order matches jobs.csv order: {seen_job_ids}")


if __name__ == "__main__":
    tests = [
        test_random_policy_many_episodes,
        test_zero_allocation_always_penalized_or_neutral,
        test_deterministic_reset_in_test_mode,
        test_capacity_enforcement_limits_allocation,
    ]
    failures = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAILED: {t.__name__}: {e}")

    if failures:
        print(f"\n{failures}/{len(tests)} tests failed")
        sys.exit(1)
    print(f"\nAll {len(tests)} tests passed")
