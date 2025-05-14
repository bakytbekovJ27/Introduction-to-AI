# Configuration settings for the Minesweeper Q-learning project
# Defines the game map and hyperparameters

MAP = [
    ".....",
    ".XXX.",
    ".....",
    ".X.X.",
    "....."
]

# Q-learning parameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPISODES = 500  # Number of training episodes
CELL_SIZE = 80  # Cell size in pixels for Pygame
FPS = 10  # Frames per second for Pygame visualization

# Miscellaneous settings
SAVE_PLOTS = True  # Flag to save learning progress plots