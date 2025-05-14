import matplotlib.pyplot as plt
import numpy as np
import pygame
from config import CELL_SIZE, SAVE_PLOTS

# Plotting class for learning progress
class Plotter:
    """
    Utility class to track and plot rewards over episodes.
    """
    def __init__(self):
        """Initialize an empty list to store rewards."""
        self.rewards = []

    def add(self, r):
        """Add a reward value to the list."""
        self.rewards.append(r)

    def show(self):
        """
        Generate and save a plot of average rewards over episodes.
        Averages are calculated over a window of 20 episodes.
        """
        window_size = 20
        averages = []
        episodes = []
        for i in range(0, len(self.rewards), window_size):
            window = self.rewards[i:i + window_size]
            if window:
                averages.append(np.mean(window))
                episodes.append(i + window_size)
        
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, averages, label='Average Reward (per 20 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Learning Progress (Averaged)')
        plt.grid(True)
        plt.legend()
        if SAVE_PLOTS:
            plt.savefig('learning_progress.png')

# Visualization constants
GRAY = (192, 192, 192)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

def draw_grid(screen, font, env, agent_pos, opened, game_over=False, elapsed_time=0):
    """
    Draw the Minesweeper grid with Pygame, showing opened cells, mines, and stats.
    - Displays opened cells count and elapsed time in the top-right corner.
    - Shows game result when game_over is True.
    """
    screen.fill(WHITE)
    for i in range(env.rows):
        for j in range(env.cols):
            pos = (i, j)
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            if pos not in opened:
                pygame.draw.rect(screen, GRAY, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
                count = env.get_mine_count(pos)
                if count > 0:
                    text = font.render(str(count), True, BLACK)
                    screen.blit(text, (j * CELL_SIZE + 30, i * CELL_SIZE + 25))
            if game_over and pos in env.mines:
                pygame.draw.rect(screen, RED, rect)
            if pos == agent_pos:
                pygame.draw.rect(screen, BLUE, rect, 4)
            pygame.draw.rect(screen, BLACK, rect, 1)
    
    # Display accuracy (opened cells) and timer
    opened_text = font.render(f"Opened: {len(opened)} / {env.total_safe}", True, BLACK)
    screen.blit(opened_text, (screen.get_width() - opened_text.get_width() - 10, 10))
    time_text = font.render(f"Time: {elapsed_time:.1f} s", True, BLACK)
    screen.blit(time_text, (screen.get_width() - time_text.get_width() - 10, 30))
    
    # Display game result
    if game_over:
        if len(opened) == env.total_safe:
            result_text = font.render("Win!", True, (0, 255, 0))  # Green for win
        else:
            result_text = font.render("Game Over", True, RED)
        screen.blit(result_text, (10, 10))
    
    pygame.display.flip()