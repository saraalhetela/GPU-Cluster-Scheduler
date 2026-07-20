"""config.py -- central configuration for data paths, RL dimensions,
cluster/reward constants, and training hyperparameters."""

DEVICE = "cpu"  # small MLP + short episodes; no GPU needed to train this agent

# --- data paths ---
PRICE_PATH = "./data/gpu_prices.csv"
JOBS_PATH = "./data/jobs.csv"
CHECKPOINT_DIR = "./checkpoints"
OUTPUTS_DIR = "./outputs"  # results.json / trace.json -- run artifacts, not inputs

# --- RL dims ---
STATE_DIM = 9  # hour_sin, hour_cos, gpu_price, job_progress, deadline_remaining,
               # gpu_hours_remaining, cluster_utilization, urgency_ratio, priority_rank

PRIORITY_RANK_VALUE = {"High": 1.0, "Medium": 0.5, "Low": 0.0}  # state feature encoding,
                        # higher = more important (distinct from baseline.py's
                        # PRIORITY_RANK, which is a sort key: lower = served first)
ACTION_DIM = 5          # 0 / 2 / 4 / 6 / 8 GPUs
GPU_ACTIONS = [0, 2, 4, 6, 8]

EPISODE_LENGTH = 24     # hours; hard cap safety backstop

# --- whole-queue simulation constants (used by baseline.py / evaluate.py's
# shared-pool path, not by GPUClusterEnv itself, which has no notion of a
# shared pool) ---
TOTAL_CLUSTER_GPUS = 32

# --- reward shaping coefficients ---
COST_COEF = 0.45
COST_URGENCY_FLOOR = 0.35  # cost sensitivity at max urgency: 1.0 (full sensitivity)
                            # at zero urgency, tapering down to this floor as the
                            # deadline approaches -- lets the agent optimize for
                            # price when it has slack, and stop caring about price
                            # once a miss is imminent.

SHAPING_COEF = 2.0       # dense per-step reward for making progress
UNMET_PENALTY_COEF = 20  # base penalty per unmet GPU-hour
PRIORITY_PENALTY_MULT = {"High": 1.6, "Medium": 1.0, "Low": 0.6}
MAX_UNMET_PENALTY = 50   # hard cap on the terminal miss penalty -- without it, a
                          # large High-priority job (up to 96 GPU-hours in this
                          # dataset) starved of capacity can produce a single
                          # transition penalty many times a normal episode's total
                          # reward, which then destabilizes training whenever that
                          # transition gets resampled from replay.

IDLE_PENALTY_COEF = 0.5  # penalty per step the job is allocated 0 GPUs
                          # while incomplete and not yet at its deadline

# --- DQN hyperparameters ---
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 5e-5
MEMORY_SIZE = 20_000
SYNC_FREQ = 300          # target network sync interval, in steps
MAX_STEPS = 30_000
EPSILON_START = 1.0
EPSILON_END = 0.05
N_STEP = 4
TRAIN_RATIO = 0.85       # splits jobs.csv 170/30 train/val

CHECKPOINT_FREQ = 1_000  # steps between checkpoint saves + validation passes
