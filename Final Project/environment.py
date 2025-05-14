import numpy as np
import random  
from config import MAP

class MinesweeperEnv:
    """
    Minesweeper environment for the Q-learning agent.
    Manages the game state, including map generation, actions, and rewards.
    """
    def __init__(self):
        """
        Initialize the environment with the map from config.py.
        Sets up states, actions, mines, and safe cells.
        """
        self.rows, self.cols = len(MAP), len(MAP[0])
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.actions = self.states.copy()
        self._generate_map()
        self.safe_cells = [s for s in self.states if s not in self.mines]

    def _generate_map(self):
        """
        Generate the game map based on MAP from config.py.
        Identifies mine locations marked with 'X'.
        """
        self.mines = []
        for i, row in enumerate(MAP):
            for j, val in enumerate(row):
                if val == 'X':
                    self.mines.append((i, j))
        self.total_safe = len(self.states) - len(self.mines)

    def reset(self):
        """
        Reset the game state for a new episode.
        Returns a random safe cell as the initial state.
        """
        self.opened = set()
        self.game_over = False
        return random.choice(self.safe_cells)

    def step(self, action):
        """
        Execute an action (open a cell) and return the next state, reward, and done flag.
        - Mine hit: -10 reward, game over.
        - Already opened: -5 reward.
        - Safe cell: 10 - 2 * adjacent_mines, +100 if all safe cells opened.
        """
        if action in self.mines:
            reward = -10
            self.game_over = True
        elif action in self.opened:
            reward = -5
        else:
            self.opened.add(action)
            mine_count = self.get_mine_count(action)
            reward = 10 - 2 * mine_count
            if len(self.opened) == self.total_safe:
                reward += 100
                self.game_over = True
        return action, reward, self.game_over

    def get_mine_count(self, pos):
        """
        Count the number of mines adjacent to the given position.
        Checks all 8 surrounding cells.
        """
        x, y = pos
        count = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.rows and 0 <= ny < self.cols and (nx, ny) in self.mines:
                    count += 1
        return count