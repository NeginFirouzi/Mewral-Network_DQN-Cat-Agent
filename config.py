# config.py
import os

# ------------------ GRID & RENDERING ------------------
GRID_SIZE = 7
CELL_PIXELS = 64
SCREEN_SIZE = GRID_SIZE * CELL_PIXELS

# ------------------ TRAINING HYPERPARAMETERS ------------------
EPISODES = 10000
MAX_STEPS = 100
BATCH_SIZE = 128
GAMMA = 0.99
LR = 1e-4
EPS_START = 2     # Full random exploration until ε decays below 1.0
EPS_END = 0.05
EPS_DECAY = 80000
TARGET_UPDATE_FREQ = 5000
REPLAY_SIZE = 10000

# ------------------ REPRODUCIBILITY ------------------
SEED = 42

# ------------------ MAP RANDOMIZATION ------------------
RANDOMIZE_MAP = False  # Toggle between static/random maps

# ------------------ STATIC MAP DEFINITIONS ------------------
OBSTACLES = {(2,2), (2,3), (3,2)}
WATER = {(5,4)}
LAVA = {(6,1)}

# ------------------ RANDOMIZATION PARAMETERS ------------------
NUM_OBSTACLES = len(OBSTACLES)
NUM_WATER = len(WATER)
NUM_LAVA = len(LAVA)

# ------------------ ASSET PATHS ------------------
ASSET_DIR = os.path.join(os.path.dirname(__file__), 'images')
CAT_IMG = "cat.png"
FOOD_IMG = "cat_food.png"
GROUND_IMG = "ground.jpg"
WALL_IMG = "wall.jpg"
WATER_IMG = "water.jpg"
LAVA_IMG = "lava.png"

# ------------------ ACTIONS ------------------
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]