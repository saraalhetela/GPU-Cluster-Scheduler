import torch

DEVICE = "cpu"  # confirmed sufficient, no GPU needed for training this agent

# --- data paths ---
PRICE_PATH = "./data/gpu_prices.csv"
JOBS_PATH = "./data/jobs.csv"
CHECKPOINT_DIR = "./checkpoints"

# --- RL dims (env-dependent, same shape as EV project) ---
STATE_DIM = 9 
PRIORITY_RANK_VALUE = {"High": 1.0, "Medium": 0.5, "Low": 0.0}  
ACTION_DIM = 5         
GPU_ACTIONS = [0, 2, 4, 6, 8]

EPISODE_LENGTH = 24     

TOTAL_CLUSTER_GPUS = 32

# --- reward shaping coefficients ---
COST_COEF = 0.45 

COST_URGENCY_FLOOR = 0.35  

SHAPING_COEF = 2.0      
UNMET_PENALTY_COEF = 20  
PRIORITY_PENALTY_MULT = {"High": 1.6, "Medium": 1.0, "Low": 0.6}
MAX_UNMET_PENALTY = 50  
IDLE_PENALTY_COEF = 0.5  

# --- algorithm-layer constants: UNCHANGED from the EV/stock-trader repo ---
# train.py imports these directly -- do not touch.
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 5e-5  
MEMORY_SIZE = 20_000
SYNC_FREQ = 600       
SYNC_FREQ = 300
MAX_STEPS = 30_000
EPSILON_START = 1.0
EPSILON_END = 0.05
N_STEP = 4
TRAIN_RATIO = 0.85   # splits jobs.csv 170/30 train/test, same pattern as fleet.csv

# --- checkpoint/validation cadence ---
CHECKPOINT_FREQ = 1_000
