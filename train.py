# train.py -- GPU cluster scheduler version
import copy
import os
import random
from collections import deque

import numpy as np
import torch

from config import *
from environment import GPUClusterEnv
from baseline import compute_hourly_demand


def train_agent(model, train_jobs, val_jobs, prices, device="cpu"):
    agent  = model
    target = copy.deepcopy(agent).to(device)
    target.load_state_dict(agent.state_dict())

    optimizer = torch.optim.RMSprop(agent.parameters(), lr=LEARNING_RATE)
    loss_fn   = torch.nn.SmoothL1Loss()

    # Replay buffer stores numpy arrays, NOT GPU tensors (cheap on RAM,
    # moved to device only at sample time). Unchanged from the stock/EV
    # version -- GPUClusterEnv's flat 7-dim state doesn't change this.
    replay     = deque(maxlen=MEMORY_SIZE)
    n_buf      = deque(maxlen=N_STEP)
    epsilon    = EPSILON_START
    step_count = 0
    last_checkpoint = 0
    train_rewards = []
    val_rewards   = []
    best_val_reward = float("-inf")
    best_step = 0

    # One env instance for the whole training run -- GPUClusterEnv handles
    # per-episode job sampling internally via reset(), unlike the stock
    # TradingEnv pattern of recreating the env every episode.
    #
    # demand_curve gives the agent a REAL scarcity signal (see
    # baseline.compute_hourly_demand) instead of the old synthetic sine
    # curve, and actually caps this episode's allocation by remaining
    # cluster headroom -- train_jobs/val_jobs each get their own curve
    # since they're different subsets of the queue with different
    # contention patterns.
    train_demand = compute_hourly_demand(train_jobs, episode_length=EPISODE_LENGTH)
    val_demand   = compute_hourly_demand(val_jobs, episode_length=EPISODE_LENGTH)

    env = GPUClusterEnv(jobs=train_jobs, prices=prices, demand_curve=train_demand, test=False)

    def flush_n(buf):
        """Compute N-step return and push numpy transition to replay."""
        if not buf:
            return
        s0, a0    = buf[0][0], buf[0][1]
        ret       = 0.0
        last_ns   = buf[-1][3]
        last_done = buf[-1][4]
        for i, (_, _, r, _, d) in enumerate(buf):
            ret += (GAMMA ** i) * r
            if d:
                last_done = True
                break
        replay.append((s0, a0, ret, last_ns, last_done))

    while step_count < MAX_STEPS:
        state = env.reset()
        done = False
        ep_reward = 0.0
        n_buf.clear()

        while not done:
            step_count += 1

            state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)

            if random.random() < epsilon:
                action = random.randint(0, ACTION_DIM - 1)
            else:
                with torch.no_grad():
                    action = agent(state_t).argmax(dim=1).item()

            # IMPORTANT: env.step() takes the raw action INDEX (0..ACTION_DIM-1)
            # and resolves it to a GPU count internally via its own ACTIONS
            # dict. Do NOT resolve it here and pass e.g. environment.ACTIONS[action]
            # in -- that double-maps the index through ACTIONS a second time
            # inside step() and silently allocates the wrong number of GPUs.
            next_state, reward, done, info = env.step(action)

            n_buf.append((state, action, reward, next_state, done))
            if len(n_buf) == N_STEP or done:
                flush_n(n_buf)
                if done:
                    while len(n_buf) > 1:
                        n_buf.popleft()
                        flush_n(n_buf)
                    n_buf.clear()

            state = next_state
            ep_reward += reward

            # ── Learning step ──────────────────────────────────────────
            if len(replay) >= BATCH_SIZE:
                batch = random.sample(replay, BATCH_SIZE)

                sb  = torch.from_numpy(np.stack([b[0] for b in batch])).float().to(device)
                ab  = torch.tensor([b[1] for b in batch], dtype=torch.long).to(device)
                rb  = torch.tensor([b[2] for b in batch], dtype=torch.float32).to(device)
                nsb = torch.from_numpy(np.stack([b[3] for b in batch])).float().to(device)
                db  = torch.tensor([b[4] for b in batch], dtype=torch.float32).to(device)

                q_pred = agent(sb).gather(1, ab.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    best_a = agent(nsb).argmax(dim=1, keepdim=True)
                    q_next = target(nsb).gather(1, best_a).squeeze(1)
                    q_tgt  = rb + (GAMMA ** N_STEP) * q_next * (1 - db)

                loss = loss_fn(q_pred, q_tgt)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=10.0)
                optimizer.step()

            if step_count % SYNC_FREQ == 0:
                target.load_state_dict(agent.state_dict())

            epsilon = max(
                EPSILON_END,
                EPSILON_START - (EPSILON_START - EPSILON_END) * step_count / MAX_STEPS
            )

            # Fixed checkpoint/val cadence: checked every step against a
            # running counter instead of `step_count % N == 0` at episode
            # boundaries only, which -- with variable-length episodes --
            # could go the entire run without ever firing.
            if step_count - last_checkpoint >= CHECKPOINT_FREQ or step_count >= MAX_STEPS:
                last_checkpoint = step_count
                os.makedirs(CHECKPOINT_DIR, exist_ok=True)
                torch.save(agent.state_dict(), f"{CHECKPOINT_DIR}/ckpt_{step_count}.pt")
                vr = _quick_val(agent, val_jobs, prices, val_demand, device)
                val_rewards.append((step_count, vr))
                if vr > best_val_reward:
                best_val_reward = vr
                best_step = step_count
                torch.save(agent.state_dict(), f"{CHECKPOINT_DIR}/ckpt_best.pt")
                print(f"Step {step_count:>7} | ε={epsilon:.3f} | "
                      f"Train ep reward={ep_reward:.2f} | Val reward={vr:.2f}")

            if step_count >= MAX_STEPS:
                break

        train_rewards.append(ep_reward)
    print(f"Best checkpoint: step {best_step} (val reward={best_val_reward:.2f}) "
          f"-> {CHECKPOINT_DIR}/ckpt_best.pt")
    return agent, train_rewards, val_rewards


def _quick_val(model, val_jobs, prices, val_demand, device, max_episodes=20):
    """Deterministic pass over up to `max_episodes` validation jobs, in
    jobs.csv order (test=True cycles jobs in order, doesn't sample) --
    averaged per-episode reward, more meaningful here than the stock
    version's single long walk since GPUClusterEnv is episodic per job.
    Uses the SAME real demand-curve/capacity-enforcement setup as
    training so validation reflects what the agent was actually trained
    under, not an easier unconstrained version of the task."""
    model.eval()
    env = GPUClusterEnv(jobs=val_jobs, prices=prices, demand_curve=val_demand, test=True)
    episodes = min(max_episodes, len(val_jobs))

    total = 0.0
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            state_t = torch.from_numpy(state).unsqueeze(0).float().to(device)
            with torch.no_grad():
                action = model(state_t).argmax(dim=1).item()
            state, reward, done, info = env.step(action)
            total += reward

    model.train()
    return total / episodes if episodes else 0.0
