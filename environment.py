"""
environment.py -- GPU cluster scheduler version.
Same filename kept (not renamed), same pattern as the EV project:
class inside fully replaced, TradingEnv/EVChargingEnv -> GPUClusterEnv,
so train.py's eventual edit stays a one-line import swap.

Scope note (matches the Master Project Definition's "Stretch Goals" list):
the CORE deliverable schedules ONE active job per episode, exactly like the
EV project scheduled one EV session per episode. Multi-job concurrent
scheduling (the agent itself reasoning about *other specific jobs*) is
still a stretch goal, not required. What CHANGED in this revision: the
agent now trains under a REAL scarcity signal instead of a purely
synthetic one -- see `demand_curve` below. That's a step toward realism,
not the same thing as joint/multi-agent scheduling.
"""

import numpy as np
import config

ACTIONS = {i: gpus for i, gpus in enumerate(config.GPU_ACTIONS)}  # {0:0,1:2,2:4,3:6,4:8}


def _synthetic_cluster_background_load(hour: float) -> float:
    """
    Fallback ONLY -- used when no demand_curve is supplied. No separate
    'rest of cluster' dataset exists, so background utilization from
    *other* jobs is modeled as a smooth synthetic curve. This is the
    original placeholder; real training now uses `demand_curve` instead
    (see baseline.compute_hourly_demand), which reflects the actual
    jobs.csv queue rather than a made-up sine wave.
    """
    return 0.25 + 0.35 * np.sin((hour - 6) / 24 * 2 * np.pi) ** 2


class GPUClusterEnv:
    def __init__(self, jobs, prices=None, demand_curve=None, total_gpus=None,
                 enforce_capacity=True, test=False):
        """
        demand_curve: optional array-like, length >= EPISODE_LENGTH, index
            = absolute hour, value = worst-case total GPU demand from the
            whole job set at that hour (see baseline.compute_hourly_demand).
            When provided, this is used BOTH for the state's
            cluster_utilization feature (a real signal instead of the old
            synthetic sine curve) AND, if enforce_capacity=True, to
            actually cap this episode's job's allocation by remaining
            cluster headroom. When None, falls back to the old
            synthetic-only behavior with no capacity constraint at all
            (backward compatible with pre-scarcity code).
        total_gpus: cluster capacity for the headroom calculation.
            Defaults to config.TOTAL_CLUSTER_GPUS.
        enforce_capacity: if False, demand_curve still informs the state
            feature but does NOT limit gpus_allocated -- used for
            isolated/diagnostic evaluation where you want the model to see
            a realistic utilization reading without the outcome being
            constrained by it.
        """
        self.jobs = jobs
        self.prices = prices if prices is not None else {}
        self.demand_curve = np.asarray(demand_curve) if demand_curve is not None else None
        self.total_gpus = total_gpus or config.TOTAL_CLUSTER_GPUS
        self.enforce_capacity = enforce_capacity
        self.test = test
        self._test_idx = 0
        self.steps_taken = 0
        self.done = True  # no reset() call here -- standard env convention is
                           # __init__ never auto-resets, the caller's first
                           # reset() call should be what starts episode 0.
                           # (This used to call self.reset() directly, which
                           # silently consumed test-mode job index 0 before
                           # the caller's first reset() ever ran -- caught by
                           # test_env.py's test_deterministic_reset_in_test_mode.)

    def reset(self):
        if self.test:
            job = self.jobs[self._test_idx % len(self.jobs)]
            self._test_idx += 1
        else:
            job = self.jobs[np.random.randint(0, len(self.jobs))]

        self.job = job
        self.hour = float(job["arrival_time"])
        self.deadline = float(job["deadline"])
        self.max_gpus = int(job["max_gpus"])
        self.gpu_hours_required = float(job["gpu_hours_required"])
        self.job_progress = float(job.get("initial_progress", 0.0))
        self.gpu_hours_remaining = self.gpu_hours_required * (1.0 - self.job_progress)
        self.steps_taken = 0
        self.done = False

        return self._get_state()

    def _demand_at(self, hour: float) -> float:
        """Worst-case total demand from the whole job set at this hour,
        indexed cyclically in case an episode's hour counter ever runs
        past len(demand_curve) (shouldn't normally happen given deadlines
        are capped within EPISODE_LENGTH, but be defensive)."""
        idx = int(hour) % len(self.demand_curve)
        return float(self.demand_curve[idx])

    def step(self, action_idx):
        assert not self.done, "step() called after episode finished"
        reward = 0.0

        requested_gpus = ACTIONS[action_idx]
        gpus_allocated = min(requested_gpus, self.max_gpus)

        if self.demand_curve is not None and self.enforce_capacity:
            # Everyone else's worst-case demand this hour = total demand
            # minus this job's own worst-case contribution (already
            # counted once in demand_curve, don't double count it).
            background_demand = max(0.0, self._demand_at(self.hour) - self.max_gpus)
            headroom = max(0.0, self.total_gpus - background_demand)
            gpus_allocated = min(gpus_allocated, headroom)

        gpu_price_now = self._price_at(self.hour)

        # work actually done this hour: capped by what's left
        gpu_hours_used = min(float(gpus_allocated), self.gpu_hours_remaining)
        progress_delta = gpu_hours_used / self.gpu_hours_required if self.gpu_hours_required > 0 else 0.0

        # --- reward terms (matches Master Project Definition's formula) ---
        cost = gpu_price_now * gpu_hours_used
        reward -= config.COST_COEF * cost
        reward += config.SHAPING_COEF * progress_delta
        if gpus_allocated == 0 and self.job_progress < 1.0:
            reward -= config.IDLE_PENALTY_COEF

        # --- state updates ---
        self.job_progress = min(1.0, self.job_progress + progress_delta)
        self.gpu_hours_remaining = max(0.0, self.gpu_hours_remaining - gpu_hours_used)
        self.hour += 1.0
        self.steps_taken += 1

        job_complete = self.job_progress >= 1.0
        past_deadline = self.hour >= self.deadline
        length_cap_hit = self.steps_taken >= config.EPISODE_LENGTH

        self.done = job_complete or past_deadline or length_cap_hit

        if self.done and not job_complete:
            # deadline missed (or length-cap safety backstop triggered) --
            # terminal penalty on whatever GPU-hours are still unmet,
            # same treatment regardless of which condition tripped it.
            reward -= config.UNMET_PENALTY_COEF * self.gpu_hours_remaining

        info = {
            "gpu_price": gpu_price_now,
            "gpus_allocated": gpus_allocated,
            "requested_gpus": requested_gpus,
            "cost": cost,
            "job_progress": self.job_progress,
            "deadline_missed": self.done and not job_complete,
        }

        return self._get_state(), reward, self.done, info

    def _price_at(self, hour: float) -> float:
        h = int(hour) % 24
        return self.prices.get(h, 1.0)

    def _get_state(self):
        h = self.hour % 24
        hour_sin = np.sin(2 * np.pi * h / 24)
        hour_cos = np.cos(2 * np.pi * h / 24)
        gpu_price = self._price_at(self.hour)
        deadline_remaining = max(0.0, self.deadline - self.hour)

        if self.demand_curve is not None:
            # Real signal: everyone else's worst-case demand / capacity.
            background_demand = max(0.0, self._demand_at(self.hour) - self.max_gpus)
            cluster_utilization = float(np.clip(background_demand / self.total_gpus, 0.0, 1.0))
        else:
            this_job_frac = (min(config.GPU_ACTIONS[-1], self.max_gpus) / config.GPU_ACTIONS[-1])
            cluster_utilization = np.clip(
                _synthetic_cluster_background_load(h) + 0.1 * this_job_frac, 0.0, 1.0
            )

        state = np.array(
            [
                hour_sin,
                hour_cos,
                gpu_price,
                self.job_progress,
                deadline_remaining,
                self.gpu_hours_remaining,
                cluster_utilization,
            ],
            dtype=np.float32,
        )
        return state