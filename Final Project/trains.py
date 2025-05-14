import os
import time
import pygame
import numpy as np
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from utils import draw_grid, Plotter
from q_learning import QLearningAgent
from environment import MinesweeperEnv
from config import CELL_SIZE, FPS, RICH_AVAILABLE

def train():
    """Обучает бота с визуализацией, панелью и автоматическим перезапуском эпизодов."""
    # Создание папок
    os.makedirs("utils/images", exist_ok=True)
    os.makedirs("utils/fonts", exist_ok=True)

    # Инициализация среды и агента
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    plotter = Plotter()

    # Настройка Pygame
    PANEL_HEIGHT = 50
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE + PANEL_HEIGHT))
    pygame.display.set_caption("Сапёр: Обучение бота")
    clock = pygame.time.Clock()
    pygame.font.init()
    try:
        font = pygame.font.Font("utils/font/font.ttf", 24)
    except pygame.error:
        print("Ошибка: Шрифт 'font.ttf' не найден. Используется системный шрифт.")
        font = pygame.font.SysFont("arial", 24)

    # Инициализация Rich
    if RICH_AVAILABLE:
        console = Console()

    # Параметры обучения
    episodes = 1000
    state = env.reset()
    start_time = time.time()
    total_reward = 0

    # Отрисовка начального состояния
    draw_grid(screen, font, env, state, env.opened, False, 0, PANEL_HEIGHT)

    for episode in range(episodes):
        while not env.game_over:
            # Выбор действия агентом
            s_idx = env.states.index(state)
            a_idx = agent.select_action(s_idx, env.opened, epsilon=0.1)
            action = env.actions[a_idx]
            next_state, reward, env.game_over = env.step(action)
            total_reward += reward

            # Обновление Q-таблицы
            ns_idx = env.states.index(next_state)
            agent.update(s_idx, a_idx, reward, ns_idx, env.game_over)
            state = next_state

            # Отрисовка
            draw_grid(screen, font, env, state, env.opened, env.game_over, time.time() - start_time, PANEL_HEIGHT)

            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    np.save("utils/train.npy", agent.Q)
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    # Перезапуск при клике на смайлик (во время эпизода)
                    if (env.cols * CELL_SIZE // 2 - 20 <= pos[0] <= env.cols * CELL_SIZE // 2 + 20 and
                        0 <= pos[1] <= PANEL_HEIGHT):
                        state = env.reset()
                        start_time = time.time()
                        total_reward = 0
                        draw_grid(screen, font, env, state, env.opened, False, 0, PANEL_HEIGHT)

            clock.tick(FPS)

        # Запись награды
        plotter.add(total_reward)

        # Вывод результатов эпизода
        elapsed_time = time.time() - start_time
        if RICH_AVAILABLE:
            table = Table(title=f"Эпизод {episode + 1}")
            table.add_column("Параметр", style="cyan")
            table.add_column("Значение", style="magenta")
            table.add_row("Статус", f"[bold {'green' if len(env.opened) == env.total_safe else 'red'}]{'Победа!' if len(env.opened) == env.total_safe else 'Проигрыш'}[/]")
            table.add_row("Награда", f"{total_reward:.1f}")
            table.add_row("Время", f"{elapsed_time:.1f} сек")
            table.add_row("Открыто клеток", f"{len(env.opened)} / {env.total_safe}")
            console.print(Panel(table, title="Итоги эпизода", border_style="bold blue"))
        else:
            print(f"Эпизод {episode + 1}: {'Победа!' if len(env.opened) == env.total_safe else 'Проигрыш'}")
            print(f"Награда: {total_reward:.1f}, Время: {elapsed_time:.1f} сек, Открыто клеток: {len(env.opened)} / {env.total_safe}")

        # Автоматический перезапуск эпизода
        time.sleep(1)  # Пауза 1 секунда для читаемости результатов
        state = env.reset()
        start_time = time.time()
        total_reward = 0
        env.game_over = False
        draw_grid(screen, font, env, state, env.opened, False, 0, PANEL_HEIGHT)

    # Сохранение модели и графика
    np.save("utils/train.npy", agent.Q)
    plotter.show()
    pygame.quit()

if __name__ == "__main__":
    train()
