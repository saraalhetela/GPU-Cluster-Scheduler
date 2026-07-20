import torch

DEVICE = "cpu"  # confirmed sufficient, no GPU needed for training this agent

# --- data paths ---
PRICE_PATH = "./data/gpu_prices.csv"
JOBS_PATH = "./data/jobs.csv"
CHECKPOINT_DIR = "./checkpoints"

# --- RL dims (env-dependent, same shape as EV project) ---
STATE_DIM = 9 # hour_sin, hour_cos, gpu_price, job_progress,
              # deadline_remaining, gpu_hours_remaining, cluster_utilization,
              # urgency_ratio, priority_rank

PRIORITY_RANK_VALUE = {"High": 1.0, "Medium": 0.5, "Low": 0.0}  # feature encoding,
                        # higher = more important (separate from baseline.py's
                        # PRIORITY_RANK, which is sort-order: lower = served first)
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
COST_COEF = 0.45  # pulled back from 0.6 -- 0.6 combined with a 0.15 floor
                  # dropped deadline_success_rate_pct to 75.0% (150/200
                  # completed, 30 missed), worse than the flat-COST_COEF=0.4
                  # baseline's 87.0%. Re-sweep from here: 0.45 / 0.5 / 0.55,
                  # reject anything that takes success_rate below ~85%.

COST_URGENCY_FLOOR = 0.35  # raised from 0.15 -- 0.15 let the agent get away
                            # with near-zero cost sensitivity relief even at
                            # moderate urgency, so it kept optimizing for cheap
                            # hours too late into each job's window. 0.35 still
                            # gives real savings at low urgency while backing
                            # off cost-consciousness sooner as slack shrinks.

SHAPING_COEF = 2.0       # dense per-step reward for making progress
UNMET_PENALTY_COEF = 20  # base penalty per unmet GPU-hour
PRIORITY_PENALTY_MULT = {"High": 1.6, "Medium": 1.0, "Low": 0.6}
MAX_UNMET_PENALTY = 50  # hard cap on the terminal miss penalty -- without
                         # this, a large High-priority job (gpu_hours_required
                         # up to 96 in this dataset) that gets starved of
                         # capacity can produce a single-transition penalty
                         # 15-20x a normal episode's reward, which then
                         # dominates the replay buffer every time it's
                         # resampled and destabilizes training (the -492,
                         # -345 spikes in the log). Missing a deadline is
                         # already unambiguously bad at this cap -- doesn't
                         # need to be arbitrarily worse to teach that.

IDLE_PENALTY_COEF = 0.5  # penalty per step the job is allocated 0 GPUs
                          # while incomplete and not yet at its deadline

# --- algorithm-layer constants: UNCHANGED from the EV/stock-trader repo ---
# train.py imports these directly -- do not touch.
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 5e-5   # halved from 1e-4 -- train ep reward was swinging
                        # wildly step to step (e.g. -95.90 sandwiched between
                        # near-zero values), classic sign the LR is too high
                        # for how sparse/spiky this reward signal is
MEMORY_SIZE = 20_000
SYNC_FREQ = 600         # doubled from 300 -- target network was chasing the
                        # online network too fast, compounding the instability
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
