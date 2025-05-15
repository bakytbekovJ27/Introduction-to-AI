import matplotlib.pyplot as plt
import numpy as np
import pygame
from config import CELL_SIZE, SAVE_PLOTS

class Plotter:
    """Класс для отслеживания и построения графика наград."""
    def __init__(self):
        self.rewards = []

    def add(self, r):
        self.rewards.append(r)

    def show(self):
        """Создает и сохраняет график средних наград за 20 эпизодов."""
        window_size = 20
        averages = [np.mean(self.rewards[i:i + window_size]) for i in range(0, len(self.rewards), window_size)]
        episodes = [i + window_size for i in range(0, len(self.rewards), window_size)]
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, averages, label='Средняя награда (за 20 эпизодов)')
        plt.xlabel('Эпизод')
        plt.ylabel('Средняя награда')
        plt.title('Прогресс обучения')
        plt.grid(True)
        plt.legend()
        if SAVE_PLOTS:
            plt.savefig('utils/train.png')

# Цвета для рамок и текста
BLUE = (0, 0, 255)
GRAY = (49, 51, 53)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

def draw_grid(screen, font, env, agent_pos, opened, game_over=False, elapsed_time=0, panel_height=0):
    """Отрисовывает сетку игры с панелью для времени, ходов и смайликом."""
    # Загрузка изображений клеток
    try:
        closed_img = pygame.image.load('utils/images/НК.png').convert_alpha()
        open_img = pygame.image.load('utils/images/ОК.png').convert_alpha()
        mine_img = pygame.image.load('utils/images/М.png').convert_alpha()
        closed_img = pygame.transform.scale(closed_img, (CELL_SIZE, CELL_SIZE))
        open_img = pygame.transform.scale(open_img, (CELL_SIZE, CELL_SIZE))
        mine_img = pygame.transform.scale(mine_img, (CELL_SIZE, CELL_SIZE))
        number_imgs = {
            i: pygame.image.load(f'utils/images/{i}.png').convert_alpha()
            for i in range(1, 8)
        }
        number_imgs = {i: pygame.transform.scale(img, (50, 50)) for i, img in number_imgs.items()}
        # Загрузка смайлика
        smile_img = pygame.image.load('utils/images/newgame.png').convert_alpha()
        smile_img = pygame.transform.scale(smile_img, (40, 40))  # Масштабирование для панели
    except pygame.error as e:
        print(f"Ошибка загрузки изображений: {e}")
        return

    # Очистка экрана
    screen.fill(WHITE)

    # Отрисовка панели сверху
    panel_rect = pygame.Rect(0, 0, env.cols * CELL_SIZE, panel_height)
    pygame.draw.rect(screen, GRAY, panel_rect)
    time_text = font.render(f"Время: {elapsed_time:.1f} с", True, RED)  # Красный текст
    moves_left = env.total_safe - len(opened)
    moves_text = font.render(f"Ходов: {moves_left}", True, RED)  # Красный текст
    screen.blit(time_text, (10, 10))
    screen.blit(moves_text, (env.cols * CELL_SIZE - moves_text.get_width() - 10, 10))
    # Отрисовка смайлика в центре панели
    screen.blit(smile_img, (env.cols * CELL_SIZE // 2 - 20, panel_height // 2 - 20))

    # Отрисовка игрового поля (смещено вниз на panel_height)
    for i in range(env.rows):
        for j in range(env.cols):
            pos = (i, j)
            rect = pygame.Rect(j * CELL_SIZE, i * CELL_SIZE + panel_height, CELL_SIZE, CELL_SIZE)
            if pos not in opened:
                screen.blit(closed_img, rect)
            else:
                screen.blit(open_img, rect)
                count = env.get_mine_count(pos)
                if count > 0:
                    try:
                        screen.blit(number_imgs[count], (j * CELL_SIZE + 15, i * CELL_SIZE + panel_height + 15))
                    except KeyError:
                        text = font.render(str(count), True, RED)  # Красный текст для резервной отрисовки
                        screen.blit(text, (j * CELL_SIZE + 30, i * CELL_SIZE + panel_height + 25))
            if game_over and pos in env.mines:
                screen.blit(mine_img, rect)
            if pos == agent_pos:
                pygame.draw.rect(screen, BLUE, rect, 4)
            pygame.draw.rect(screen, GRAY, rect, 1)

    pygame.display.flip()