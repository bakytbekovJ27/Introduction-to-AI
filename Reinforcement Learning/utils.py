import matplotlib.pyplot as plt
import numpy as np
import pygame
from config import CELL_SIZE

# Класс для построения графиков
class Plotter:
    def __init__(self):
        self.rewards = []

    def add(self, r):
        self.rewards.append(r)

    def show(self):
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
        plt.show()

# Функция визуализации
GRAY = (192, 192, 192)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# def draw_grid(screen, font, env, agent_pos, opened, game_over=False):
    # screen.fill(WHITE)
    # for i in range(env.rows):
    #     for j in range(env.cols):
    #         pos = (i, j)
    #         rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE)
    #         if pos not in opened:
    #             pygame.draw.rect(screen, GRAY, rect)
    #         else:
    #             pygame.draw.rect(screen, WHITE, rect)
    #             count = env.get_mine_count(pos)
    #             if count > 0:
    #                 text = font.render(str(count), True, BLACK)
    #                 screen.blit(text, (j * CELL_SIZE + 30, i * CELL_SIZE + 25))
    #         if game_over and pos in env.mines:
    #             pygame.draw.rect(screen, RED, rect)
    #         if pos == agent_pos:
    #             pygame.draw.rect(screen, BLUE, rect, 4)
    #         pygame.draw.rect(screen, BLACK, rect, 1)
    # pygame.display.flip()