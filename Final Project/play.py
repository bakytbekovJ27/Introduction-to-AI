import os
import time
import pygame
import numpy as np
import random

from config import CELL_SIZE, FPS, MAP
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
RICH_AVAILABLE = True

# Класс среды Minesweeper
class MinesweeperEnv:
    def __init__(self):
        self.rows, self.cols = len(MAP), len(MAP[0])
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.actions = self.states.copy()
        self._generate_map()
        self.safe_cells = [s for s in self.states if s not in self.mines]

    def _generate_map(self):
        self.mines = []
        for i, row in enumerate(MAP):
            for j, val in enumerate(row):
                if val == 'X':
                    self.mines.append((i, j))
        self.total_safe = len(self.states) - len(self.mines)

    def reset(self):
        self.opened = set()
        self.game_over = False
        initial_safe_cell = random.choice(self.safe_cells)
        self.opened.add(initial_safe_cell)
        return initial_safe_cell

    def step(self, action):
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

# Класс Q-learning агента
class QLearningAgent:
    def __init__(self, states, actions):
        self.states = states
        self.actions = actions
        self.Q = np.zeros((len(states), len(actions)))

    def select_action(self, state_idx, opened, epsilon=0):
        available_actions = [i for i, action in enumerate(self.actions) if action not in opened]
        if not available_actions:
            return np.random.randint(len(self.actions))
        q_values = [self.Q[state_idx, a] for a in available_actions]
        return available_actions[np.argmax(q_values)]

    def update(self, state, action, reward, next_state, done):
        q_current = self.Q[state, action]
        q_next = np.max(self.Q[next_state]) if not done else 0
        self.Q[state, action] = q_current + 0.1 * (reward + 0.9 * q_next - q_current)

# Константы для визуализации
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
        smile_img = pygame.transform.scale(smile_img, (40, 40))
    except pygame.error as e:
        print(f"Ошибка загрузки изображений: {e}")
        return

    # Очистка экрана
    screen.fill(WHITE)

    # Отрисовка панели сверху
    panel_rect = pygame.Rect(0, 0, env.cols * CELL_SIZE, panel_height)
    pygame.draw.rect(screen, GRAY, panel_rect)
    time_text = font.render(f"Время: {elapsed_time:.1f} с", True, RED)
    moves_left = env.total_safe - len(opened)
    moves_text = font.render(f"Ходов: {moves_left}", True, RED)
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
                        text = font.render(str(count), True, RED)
                        screen.blit(text, (j * CELL_SIZE + 30, i * CELL_SIZE + panel_height + 25))
            if game_over and pos in env.mines:
                screen.blit(mine_img, rect)
            if pos == agent_pos:
                pygame.draw.rect(screen, BLUE, rect, 4)
            pygame.draw.rect(screen, GRAY, rect, 1)

    # Результат игры
    if game_over:
        result_text = font.render("Победа!" if len(opened) == env.total_safe else "Игра окончена", True, RED)
        screen.blit(result_text, (10, panel_height + 10))

    pygame.display.flip()

def play_human_vs_model():
    """Запускает игру 'Человек против бота' с панелью, смайликом и автоматическим перезапуском."""
    # Создание папок
    os.makedirs("utils/images", exist_ok=True)
    os.makedirs("utils/font", exist_ok=True)

    # Инициализация среды и агента
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    try:
        agent.Q = np.load("utils/final_model.npy")
    except FileNotFoundError:
        print("Ошибка: Файл модели 'utils/final_model.npy' не найден.")
        return

    # Настройка Pygame
    PANEL_HEIGHT = 50
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE + PANEL_HEIGHT))
    pygame.display.set_caption("Сапёр: Человек против бота")
    clock = pygame.time.Clock()
    pygame.font.init()
    try:
        font = pygame.font.Font("utils/fonts/font.ttf", 24)
    except pygame.error:
        print("Ошибка: Шрифт 'font.ttf' не найден. Используется системный шрифт.")
        font = pygame.font.SysFont("arial", 24)

    # Инициализация Rich
    if RICH_AVAILABLE:
        console = Console()

    # Начальное состояние
    state = env.reset()
    human_turn = False  # Бот ходит первым
    start_time = time.time()
    last_player = None

    # Отрисовка начального состояния
    draw_grid(screen, font, env, state, env.opened, False, 0, PANEL_HEIGHT)

    while True:  # Бесконечный цикл для перезапуска
        while not env.game_over:
            if human_turn:
                # Ход человека: выбор клетки кликом или перезапуск через смайлик
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        pos = pygame.mouse.get_pos()
                        # Проверка клика на смайлик
                        if (env.cols * CELL_SIZE // 2 - 20 <= pos[0] <= env.cols * CELL_SIZE // 2 + 20 and
                            0 <= pos[1] <= PANEL_HEIGHT):
                            state = env.reset()
                            start_time = time.time()
                            last_player = None
                            human_turn = False
                            draw_grid(screen, font, env, state, env.opened, False, 0, PANEL_HEIGHT)
                        # Клик на игровое поле
                        col, row = pos[0] // CELL_SIZE, (pos[1] - PANEL_HEIGHT) // CELL_SIZE
                        action = (row, col)
                        if action in env.states and action not in env.opened:
                            state, reward, env.game_over = env.step(action)
                            last_player = "человек"
                            draw_grid(screen, font, env, action, env.opened, env.game_over, time.time() - start_time, PANEL_HEIGHT)
                            human_turn = False
            else:
                # Ход бота
                s_idx = env.states.index(state)
                a_idx = agent.select_action(s_idx, env.opened)
                action = env.actions[a_idx]
                state, reward, env.game_over = env.step(action)
                last_player = "бот"
                draw_grid(screen, font, env, action, env.opened, env.game_over, time.time() - start_time, PANEL_HEIGHT)
                human_turn = True

            clock.tick(FPS)

        # Итог игры
        winner = last_player if len(env.opened) == env.total_safe else ("человек" if last_player == "бот" else "бот")
        loser = "бот" if winner == "человек" else "человек"
        result = "Победа!" if len(env.opened) == env.total_safe else "Игра окончена"
        elapsed_time = time.time() - start_time

        if RICH_AVAILABLE:
            table = Table(title="Результат игры")
            table.add_column("Параметр", style="cyan")
            table.add_column("Значение", style="magenta")
            table.add_row("Статус", f"[bold {'green' if result == 'Победа!' else 'red'}]{result}[/]")
            table.add_row("Победитель", winner.capitalize())
            table.add_row("Проигравший", loser.capitalize())
            table.add_row("Время игры", f"{elapsed_time:.1f} сек")
            table.add_row("Открыто клеток", f"{len(env.opened)} / {env.total_safe}")
            console.print(Panel(table, title="Итоги", border_style="bold blue"))
        else:
            print(f"{result} Выиграл: {winner.capitalize()}. Проигравший: {loser.capitalize()}.")
            print(f"Время игры: {elapsed_time:.1f} сек, Открыто клеток: {len(env.opened)} / {env.total_safe}")

        # Автоматический перезапуск
        time.sleep(1)  # Пауза 1 секунда для читаемости результатов
        state = env.reset()
        start_time = time.time()
        last_player = None
        env.game_over = False
        human_turn = False
        draw_grid(screen, font, env, state, env.opened, False, 0, PANEL_HEIGHT)

    pygame.quit()

if __name__ == "__main__":
    play_human_vs_model()