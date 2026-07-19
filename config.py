import torch

DEVICE = "cpu"  # confirmed sufficient, no GPU needed for training this agent

# --- data paths ---
PRICE_PATH = "./data/gpu_prices.csv"
JOBS_PATH = "./data/jobs.csv"
CHECKPOINT_DIR = "./checkpoints"

# --- RL dims (env-dependent, same shape as EV project) ---
STATE_DIM = 8 # hour_sin, hour_cos, gpu_price, job_progress,
              # deadline_remaining, gpu_hours_remaining, cluster_utilization,
              # urgency_ratio
ACTION_DIM = 5          # 0 / 2 / 4 / 6 / 8 GPUs
GPU_ACTIONS = [0, 2, 4, 6, 8]

EPISODE_LENGTH = 24     # hours, hard cap safety backstop (mirrors EV project's cap)

# --- baseline.py: whole-queue simulation, NOT used by GPUClusterEnv itself ---
# GPUClusterEnv schedules one job per episode and has no notion of a shared
# pool; baseline.py's FCFS/Always-Max/Priority comparison runs the *entire*
# jobs.csv queue against a fixed-size cluster instead, which is where this
# constant is actually used.
TOTAL_CLUSTER_GPUS = 32

# --- reward shaping coefficients ---
COST_COEF = 0.6  # bumped from 0.4 -- now safe to raise since cost penalty is
                 # urgency-scaled (see COST_URGENCY_FLOOR below), so it no
                 # longer competes flatly against SHAPING_COEF/UNMET_PENALTY_COEF
                 # the way a flat higher COST_COEF would have. Still worth
                 # sweeping 0.6 / 0.8 / 1.0 and checking deadline_success_rate_pct
                 # doesn't regress below ~85%.

COST_URGENCY_FLOOR = 0.15  # at max urgency_ratio (job in real trouble, no
                            # slack left), cost penalty shrinks to this
                            # fraction of COST_COEF -- lets the agent pay up to
                            # finish without being punished for it. At zero
                            # urgency (lots of slack), full COST_COEF applies,
                            # so the agent has real incentive to wait for a
                            # cheaper hour instead of paying peak price.

SHAPING_COEF = 2.0       # dense per-step reward for making progress
UNMET_PENALTY_COEF = 20  # terminal penalty per unmet GPU-hour (same value that
                          # worked for the EV project's unmet_energy penalty)
IDLE_PENALTY_COEF = 0.5  # penalty per step the job is allocated 0 GPUs
                          # while incomplete and not yet at its deadline

# --- algorithm-layer constants: UNCHANGED from the EV/stock-trader repo ---
# train.py imports these directly -- do not touch.
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-4
MEMORY_SIZE = 20_000
SYNC_FREQ = 300
MAX_STEPS = 30_000
EPSILON_START = 1.0
EPSILON_END = 0.05
N_STEP = 4
TRAIN_RATIO = 0.85   # splits jobs.csv 170/30 train/test, same pattern as fleet.csv

# --- checkpoint/validation cadence ---
# Added for the GPU pivot: the stock/EV version hardcoded "every 5000
# steps," which only ever fires once now that MAX_STEPS itself is 5000.
# train.py checks a running counter (step_count - last_checkpoint) against
# this instead of a modulo at episode boundaries, so it actually fires
# reliably regardless of variable episode length.
CHECKPOINT_FREQ = 1_000
