# GPU Cluster Scheduler — RL Agent

An RL agent that learns to schedule GPU allocation for AI training/inference
jobs on a shared, capacity-constrained cluster, trained under a real
scarcity signal derived from actual cluster job traces.

## Problem

GPU clusters are chronically oversubscribed: every hour of the day sees
worst-case demand well above available capacity (32 GPUs in this setup,
against per-hour demand ranging ~40–216 GPUs). Simple scheduling heuristics
either serve jobs in arrival order or by a fixed priority tier, with no
learned sense of *when* to allocate more or less GPU to a given job based
on remaining time, remaining work, and how contended the cluster is right
now.

## Solution

A Dueling DQN agent (`DuelingMLP`) learns, per job, how many GPUs to
request each hour — trading off job cost (GPU-price × GPU-hours used)
against the risk of missing the job's deadline. It's trained one job per
episode, under a real per-hour scarcity signal computed from the actual
job queue (`compute_hourly_demand`), so the state it sees during training
matches the state it sees at evaluation time.

## Approach

- **Environment**: `GPUClusterEnv` — one job per episode, 7-dim state
  (hour sin/cos, GPU price, job progress, deadline remaining, GPU-hours
  remaining, cluster utilization), 5 discrete GPU-allocation actions
  (0/2/4/6/8 GPUs).
- **Scarcity signal**: `compute_hourly_demand` — a static, cheaply
  precomputed proxy for contention: per hour, the sum of every arrived,
  not-yet-deadlined job's `max_gpus`, assuming none finish early. This
  both feeds the state's `cluster_utilization` feature and caps the
  agent's own allocation by remaining headroom during training.
- **Training**: N-step Double DQN, dueling value/advantage heads, replay
  buffer, target network — 5,000 steps, 170/30 train/val job split.
- **Evaluation**: two consistent paths, both driven off the same real
  demand curve —
  1. **Shared-pool simulation** (the headline number): the *entire* job
     queue runs hour-by-hour against one real 32-GPU pool, competing
     against FCFS / Always-Max / Priority baselines under identical
     conditions.
  2. **Isolated per-job diagnostic**: each job evaluated alone, at full
     priority — useful for sanity-checking the policy itself, but
     explicitly *not* comparable to the baselines and excluded from
     reporting (see results below).

## Results

Same shared 32-GPU pool, same job queue, for every policy:

| Policy | Utilization | Cost | Completed | Missed | Success Rate |
|---|---|---|---|---|---|
| FCFS | 86.3% | $627.52 | 125 | 53 | 62.5% |
| Always Max | 86.3% | $627.52 | 125 | 53 | 62.5% |
| Priority | 86.1% | $626.11 | 141 | 38 | 70.5% |
| **RL Agent** | **81.7%** | **$605.43** | **157** | **22** | **78.5%** |

**+8.0 points deadline success rate over the best baseline (Priority),
at $20.68 *lower* cost and *lower* utilization.** The agent isn't just
completing more jobs — it's doing so more cheaply and with less GPU time
consumed overall, by being more selective about which jobs to serve and
when, rather than allocating maximally whenever a slot is free.

*(An isolated, no-capacity-constraint diagnostic run shows 85.5% success —
this number is intentionally excluded from the comparison above, since it
isn't under the same shared-pool contention the baselines are evaluated
under. It's only useful as a sanity check that the underlying policy is
sound.)*

## Honesty / Scope Notes

**Real vs. synthetic data.** `data/jobs.csv` is a hybrid:

| Source | Count | % |
|---|---|---|
| Real (Alibaba PAI GPU-cluster trace) | 170 | 85.0% |
| Synthetic (random) | 20 | 10.0% |
| Synthetic (guaranteed edge case) | 10 | 5.0% |

The synthetic edge cases are forced tight-deadline/high-priority jobs,
included so the reward function's penalty terms are demonstrably
exercised in the demo rather than left to chance.

Also worth stating plainly: `deadline` and `priority` are **not** real
fields — neither real-data source used (Philly, Alibaba PAI) logs an SLA
or priority field. Deadlines are derived via a slack multiplier off each
job's minimum feasible completion time, and priority is derived from a
hash of a real grouping field (virtual cluster / user), bucketed Low/
Medium/High. These are principled proxies, not ground-truth labels — say
so if asked.

**Scarcity-training scope.** The agent learns per-job GPU allocation
under a real, but approximate, scarcity signal — `compute_hourly_demand`
is a worst-case static proxy (assumes no job finishes early), not a live
multi-agent simulation. The agent does not know about *other specific
jobs* in the queue and does not learn to yield GPUs to a more urgent one.
When multiple jobs compete for the same hour's capacity in the shared-pool
evaluation, an earliest-deadline-first rule arbitrates between them — this
arbitration is a necessary bolt-on to run a single-job-trained policy in a
multi-job setting, not something the agent itself learned. True joint /
multi-agent scheduling is out of scope for this project.

## Repo Layout

```
data_generation.py       synthetic job/price generation
real_data.py              real (Philly/Alibaba) trace parsing + hybrid build
data_preprocessing.py     loads jobs.csv / gpu_prices.csv for the env
environment.py            GPUClusterEnv
model.py                  DuelingMLP
baseline.py               FCFS / Always-Max / Priority heuristics + compute_hourly_demand
train.py                  DQN training loop
evaluate.py               shared-pool + isolated evaluation
main.py                   full pipeline: train -> evaluate -> results.json
test_env.py               environment unit tests
```

## Running It

```
python main.py
```

Trains from scratch on `data/jobs.csv`, evaluates against all three
baselines on the shared 32-GPU pool, and writes `results.json` +
training/validation plots to `plots/`.