import os
import time
import pygame
import numpy as np

from rich.panel import Panel
from rich.table import Table
from rich.console import Console

from utils import draw_grid
from q_learning import QLearningAgent
from environment import MinesweeperEnv
from config import CELL_SIZE, FPS, RICH_AVAILABLE

def play_human_vs_model():
    """Запускает игру 'Человек против бота' с панелью для времени и ходов и смайликом для перезапуска."""
    # Создание папок для изображений и шрифтов
    os.makedirs("utils/images", exist_ok=True)
    os.makedirs("utils/fonts", exist_ok=True)

    # Инициализация среды и агента
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    try:
        agent.Q = np.load("utils/train.npy")
    except FileNotFoundError:
        print("Ошибка: Файл модели 'utils/train.npy' не найден.")
        return

    # Настройка Pygame с дополнительной высотой для панели
    PANEL_HEIGHT = 50
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE + PANEL_HEIGHT))
    pygame.display.set_caption("Сапёр: Человек против бота")
    clock = pygame.time.Clock()
    pygame.font.init()  # Инициализация модуля шрифтов
    try:
        font = pygame.font.Font("utils/font/font.ttf", 24)  # Загрузка пользовательского шрифта
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
                        # Проверка клика на смайлик (центр панели)
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
                # Ход бота: выбор действия из Q-таблицы
                s_idx = env.states.index(state)
                a_idx = agent.select_action(s_idx, env.opened, epsilon=0)
                action = env.actions[a_idx]
                state, reward, env.game_over = env.step(action)
                last_player = "бот"
                draw_grid(screen, font, env, action, env.opened, env.game_over, time.time() - start_time, PANEL_HEIGHT)
                human_turn = True

            # Итог игры
            if env.game_over:
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

            clock.tick(FPS)

        # Ожидание клика на смайлик для перезапуска
        while env.game_over:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    if (env.cols * CELL_SIZE // 2 - 20 <= pos[0] <= env.cols * CELL_SIZE // 2 + 20 and
                        0 <= pos[1] <= PANEL_HEIGHT):
                        state = env.reset()
                        start_time = time.time()
                        last_player = None
                        env.game_over = False
                        human_turn = False
                        draw_grid(screen, font, env, state, env.opened, False, 0, PANEL_HEIGHT)
                        break

if __name__ == "__main__":
    play_human_vs_model()
