# GPU Cluster Scheduler — RL Agent

> An RL agent that learns to schedule GPU allocation for AI training/inference jobs on a shared, capacity-constrained cluster, trained under a real scarcity signal derived from actual cluster job traces.


## Problem

GPU clusters are chronically oversubscribed: every hour of the day sees
worst-case demand well above available capacity (32 GPUs in this setup,
against per-hour demand ranging ~40–216 GPUs). Simple scheduling heuristics
either serve jobs in arrival order or by a fixed priority tier, with no
learned sense of *when* to allocate more or less GPU to a given job based
on remaining time, remaining work, and how contended the cluster is right
now.

---

## Solution

A Dueling DQN agent (`DuelingMLP`) learns, per job, how many GPUs to
request each hour — trading off job cost (GPU-price × GPU-hours used)
against the risk of missing the job's deadline. It's trained one job per
episode, under a real per-hour scarcity signal computed from the actual
job queue (`compute_hourly_demand`), so the state it sees during training
matches the state it sees at evaluation time.

---

## Approach

### Environment  
**`GPUClusterEnv`** — one job per episode, 9-dim state
  (hour sin/cos, GPU price, job progress, deadline remaining, GPU-hours
  remaining, cluster utilization, urgency ratio, priority rank), 5 discrete
  GPU-allocation actions (0/2/4/6/8 GPUs).
### Scarcity signal
**`compute_hourly_demand`** — a static, cheaply
  precomputed proxy for contention: per hour, the sum of every arrived,
  not-yet-deadlined job's `max_gpus`, assuming none finish early. This
  both feeds the state's `cluster_utilization` feature and caps the
  agent's own allocation by remaining headroom during training.
- **Training**: N-step Double DQN, dueling value/advantage heads, replay
  buffer, target network — 30,000 steps, 170/30 train/val job split.
### Evaluation
two consistent paths, both driven off the same real
  demand curve —
  1. **Shared-pool simulation** (the headline number): the *entire* job
     queue runs hour-by-hour against one real 32-GPU pool, competing
     against FCFS / Always-Max / Priority baselines under identical
     conditions.
  2. **Isolated per-job diagnostic**: each job evaluated alone, at full
     priority — useful for sanity-checking the policy itself, but
     explicitly *not* comparable to the baselines and excluded from
     reporting (see results below).

---

## Results

Same shared 32-GPU pool, same job queue, for every policy:

| Policy | Utilization | Cost | Completed | Missed | Success Rate |
|:-------|------------:|-----:|----------:|-------:|-------
| FCFS | 86.3% | $627.52 | 125 | 53 | 62.5% |
| Always Max | 86.3% | $627.52 | 125 | 53 | 62.5% |
| Priority | 86.1% | $626.11 | 141 | 38 | 70.5% |
| **RL Agent** | **83.2%** | **$603.22** | **161** | **18** | **80.5%** |


> **+10.0 points deadline success rate over the best baseline (Priority), at $22.89 lower cost and lower utilization.**
>
> The agent isn't just
completing more jobs — it's doing so more cheaply and with less GPU time
consumed overall, by being more selective about which jobs to serve and
when, rather than allocating maximally whenever a slot is free.

*(An isolated, no-capacity-constraint diagnostic run shows 82.5% success —
this number is intentionally excluded from the comparison above, since it
isn't under the same shared-pool contention the baselines are evaluated
under. It's only useful as a sanity check that the underlying policy is
sound.)*

---

## Training Diagnostics

`main.py` writes two plots to `plots/` on every run:

| Training episode reward | Validation reward |
|---|---|
|<p align="center"> <img width="3000" height="1500" alt="train_rewards" src="https://github.com/user-attachments/assets/229c242e-d77d-449d-917a-e43d78d2103f" /> | <img width="3000" height="1500" alt="val_rewards" src="https://github.com/user-attachments/assets/f0b477e9-1b65-4c82-9cf6-14cc8c29ae9f" /> |</p>

Per-episode training reward is noisy by design — each episode is a single
job with its own deadline pressure and priority, so reward swings between
roughly +2 (an easy, comfortably-met job) and the −50 penalty cap (a large
High-priority job that got starved of capacity). Validation reward
(averaged over 20 held-out jobs every 1,000 steps) settles into a stable
band around −1.2 with occasional dips where epsilon-driven exploration or
a batch of harder validation jobs temporarily pulls the average down —
the best checkpoint (`ckpt_best.pt`, used for all reported results) is
selected from this curve, not from the final training step.

---

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
fields — the real-data source used (Alibaba PAI) logs no SLA
or priority field. Deadlines are derived via a slack multiplier off each
job's minimum feasible completion time, and priority is derived from a
hash of a real grouping field (virtual cluster / user), bucketed Low/
Medium/High. These are principled proxies, not ground-truth labels.

**Scarcity-training scope.** The agent learns per-job GPU allocation
under a real, but approximate, scarcity signal — `compute_hourly_demand`
is a worst-case static proxy (assumes no job finishes early), not a live
multi-agent simulation. The agent does not know about *other specific
jobs* in the queue and does not learn to yield GPUs to a more urgent one.
When multiple jobs compete for the same hour's capacity in the shared-pool
evaluation, a priority-then-deadline rule arbitrates between them — this
arbitration is a necessary bolt-on to run a single-job-trained policy in a
multi-job setting, not something the agent itself learned. True joint /
multi-agent scheduling is out of scope for this project.

---

## Repo Layout

```
.
├── data/                     jobs.csv + gpu_prices.csv -- committed, ready to run against               
├── data_generation.py        synthetic job/price generation
├── real_data.py               real (Alibaba PAI) trace parsing + hybrid build
├── data_preprocessing.py      loads data/jobs.csv / data/gpu_prices.csv for the env
├── environment.py             GPUClusterEnv
├── model.py                   DuelingMLP
├── baseline.py                FCFS / Always-Max / Priority heuristics + compute_hourly_demand
├── train.py                   DQN training loop
├── evaluate.py                shared-pool + isolated evaluation
├── export_trace.py            hour-by-hour trace export for the dashboard demo -> outputs/trace.json
├── main.py                    full pipeline: train -> evaluate -> outputs/results.json
├── test_env.py                environment unit tests 
└── dashboard_demo.html        animated hour-by-hour dashboard (reads outputs/trace.json)
```

---

## Data provenance

`data/jobs.csv` (200 rows, ~20 KB) and `data/gpu_prices.csv` are committed
directly so the repo runs out of the box -- clone it, `python main.py`,
done. They're fully derived, small, and don't carry the raw trace data's
own licensing/hosting concerns.

The raw trace itself (Alibaba's `pai_job_table.csv` / `pai_task_table.csv`)
is *not* committed -- it's a large external download from Alibaba's own
host. If you want to see exactly
how `data/jobs.csv` was built, or rebuild it from a fresher trace pull,
`real_data.py`'s module docstring has the download commands and
`python real_data.py build` reproduces the hybrid file end to end (see
`real_data.py` for the full pipeline, or `data_generation.py` for the
synthetic-only fallback).

---

## Running It

Clone the repository and run:
```
python main.py
```

Trains from scratch on `data/jobs.csv`, evaluates against all three
baselines on the shared 32-GPU pool, writes `outputs/results.json` +
`outputs/trace.json`, and serves `dashboard_demo.html` locally so the run
can be watched hour by hour.
