import os
import time
import pygame
import logging
import imageio
import asyncio
import platform
import numpy as np
import matplotlib.pyplot as plt
from utils import Plotter, draw_grid
from q_learning import QLearningAgent
from environment import MinesweeperEnv
from config import EPISODES, CELL_SIZE, FPS, SAVE_PLOTS, IMAGEIO_AVAILABLE

# Создаем директорию utils, если ее не существует
os.makedirs("utils", exist_ok=True)

# Настройка логирования: вывод в консоль и сохранение в файл utils/train.log
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('utils/train.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Глобальные переменные для окружения, агента, визуализации и записи GIF
env = None
agent = None
screen = None
font = None
clock = None
plotter = None
gif_frames = None

def setup():
    """Инициализация окружения, агента, Pygame и настройки записи GIF."""
    global env, agent, screen, font, clock, plotter, gif_frames
    # Создаем игровое окружение и агента для обучения Q-learning
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    plotter = Plotter()
    
    # Инициализация Pygame для визуализации игрового процесса
    pygame.init()
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE))
    pygame.display.set_caption("Minesweeper: Q-Learning Training")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)
    
    # Настройка записи GIF, если библиотека imageio доступна
    if IMAGEIO_AVAILABLE:
        gif_frames = []
        logger.info("Запись GIF включена: utils/train.gif")
    else:
        logger.warning("Запись GIF отключена: imageio не установлен")

async def train():
    """Запуск обучения Q-learning с визуализацией и сохранением результатов."""
    global env, agent, screen, font, clock, plotter, gif_frames
    EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.0001, 0.004
    logger.info("Начало обучения")

    for ep in range(1, EPISODES + 1):
        # Расчет значения epsilon для текущего эпизода
        epsilon = max(EPSILON_END, EPSILON_START - (ep - 1) * EPSILON_DECAY)
        state = env.reset()
        total_reward, step, done = 0, 0, False
        start_time = time.time()

        while not done:
            # Обработка событий Pygame (например, закрытие окна)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Обучение прервано пользователем")
                    if IMAGEIO_AVAILABLE and gif_frames:
                        imageio.mimsave('utils/train.gif', gif_frames, duration=1000/FPS)
                        logger.info("GIF сохранен в utils/train.gif")
                    pygame.quit()
                    return

            # Выбор действия агентом на основе текущего состояния и epsilon
            s_idx = env.states.index(state)
            a_idx = agent.select_action(s_idx, env.opened, epsilon)
            action = env.actions[a_idx]
            # Выполнение действия, получение следующего состояния, награды и статуса окончания эпизода
            next_state, reward, done = env.step(action)
            next_s_idx = env.states.index(next_state)
            # Обновление Q-таблицы
            agent.update(s_idx, a_idx, reward, next_s_idx, done)

            logger.info(f"Шаг {step}: Состояние={state}, Действие={action}, Награда={reward}")
            elapsed_time = time.time() - start_time
            # Отрисовка игрового поля
            draw_grid(screen, font, env, action, env.opened, done, elapsed_time)

            # Сохранение кадров для GIF каждые 10 эпизодов, если imageio доступен
            if IMAGEIO_AVAILABLE and ep % 10 == 0:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))
                gif_frames.append(frame)

            clock.tick(FPS)
            total_reward += reward
            state = next_state
            step += 1
            await asyncio.sleep(1.0 / FPS)

        logger.info(f"Эпизод {ep} завершен. Общая награда: {total_reward}, EPSILON: {epsilon:.3f}")
        # Добавляем итоговую награду эпизода для построения графика
        plotter.add(total_reward)

    # Сохранение Q-таблицы в файл utils/train.npy
    np.save("utils/train.npy", agent.Q)
    logger.info("Q-таблица сохранена в utils/train.npy")
    
    # Сохранение GIF, если кадры были записаны
    if IMAGEIO_AVAILABLE and gif_frames:
        imageio.mimsave('utils/train.gif', gif_frames, duration=1000/FPS)
        logger.info("GIF сохранен в utils/train.gif")
    
    # Сохранение графика результатов обучения, если включено сохранение графиков
    if SAVE_PLOTS:
        plt.figure(figsize=(10, 6))
        window_size = 20
        averages = [np.mean(plotter.rewards[i:i + window_size]) for i in range(0, len(plotter.rewards), window_size)]
        episodes = [i + window_size for i in range(0, len(plotter.rewards), window_size)]
        plt.plot(episodes, averages, label='Average Reward (per 20 episodes)')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.title('Learning Progress')
        plt.grid(True)
        plt.legend()
        plt.savefig('utils/train.png')
        logger.info("График сохранен в utils/train.png")
    
    pygame.quit()
    logger.info("Обучение завершено")

async def main():
    """Основная асинхронная функция для запуска обучения."""
    setup()
    await train()

# Если запускаем на Emscripten, используем asyncio.ensure_future, иначе через asyncio.run
if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())