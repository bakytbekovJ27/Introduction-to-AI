import asyncio
import platform
import pygame
import time
import numpy as np
from config import EPISODES, CELL_SIZE, FPS, MAP, ALPHA, GAMMA, SAVE_PLOTS
import matplotlib.pyplot as plt
import random
import logging

# Попытка импорта imageio с обработкой ошибки
try:
    import imageio
    IMAGEIO_AVAILABLE = True
except ImportError:
    IMAGEIO_AVAILABLE = False
    print("Ошибка: Не удалось импортировать imageio. Запись GIF будет отключена. Установите imageio: pip install imageio")

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),  # Логи в файл
        logging.StreamHandler()  # Логи в консоль
    ]
)
logger = logging.getLogger(__name__)

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
        return random.choice(self.safe_cells)

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

    def select_action(self, state_idx, opened, epsilon):
        available_actions = [i for i, action in enumerate(self.actions) if action not in opened]
        if not available_actions:
            return np.random.randint(len(self.actions))
        if np.random.rand() < epsilon:
            return np.random.choice(available_actions)
        q_values = [self.Q[state_idx, a] for a in available_actions]
        return available_actions[np.argmax(q_values)]

    def update(self, s_idx, a_idx, reward, next_s_idx, done):
        target = reward
        if not done:
            target += GAMMA * np.max(self.Q[next_s_idx])
        self.Q[s_idx, a_idx] += ALPHA * (target - self.Q[s_idx, a_idx])

# Класс для построения графика прогресса
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
        plt.plot(episodes, averages, label='Средняя награда (за 20 эпизодов)')
        plt.xlabel('Эпизод')
        plt.ylabel('Средняя награда')
        plt.title('Прогресс обучения (усреднённый)')
        plt.grid(True)
        plt.legend()
        if SAVE_PLOTS:
            plt.savefig('learning_progress.png')

# Константы для визуализации
GRAY = (192, 192, 192)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

def draw_grid(screen, font, env, agent_pos, opened, game_over=False, elapsed_time=0):
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
    
    opened_text = font.render(f"Открыто: {len(opened)} / {env.total_safe}", True, BLACK)
    screen.blit(opened_text, (screen.get_width() - opened_text.get_width() - 10, 10))
    time_text = font.render(f"Время: {elapsed_time:.1f} с", True, BLACK)
    screen.blit(time_text, (screen.get_width() - time_text.get_width() - 10, 30))
    
    if game_over:
        if len(opened) == env.total_safe:
            result_text = font.render("Победа!", True, (0, 255, 0))
        else:
            result_text = font.render("Игра окончена", True, RED)
        screen.blit(result_text, (10, 10))
    
    pygame.display.flip()

def log_step(step, state, action, reward):
    logger.info(f"Шаг {step}: Состояние={state}, Действие={action}, Награда={reward}")

# Глобальные переменные
env = None
agent = None
screen = None
font = None
clock = None
plotter = None
gif_frames = None

def setup():
    global env, agent, screen, font, clock, plotter, gif_frames
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    plotter = Plotter()

    pygame.init()
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE))
    pygame.display.set_caption("Сапёр: Обучение Q-Learning")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    # Настройка для записи GIF
    if IMAGEIO_AVAILABLE:
        gif_frames = []
        logger.info("Запись GIF настроена: training.gif")
    else:
        logger.warning("Запись GIF отключена: imageio не установлен")

async def train():
    global env, agent, screen, font, clock, plotter, gif_frames
    
    EPSILON_START = 1.0
    EPSILON_END = 0.0001
    EPSILON_DECAY = 0.004

    logger.info("Начало обучения")
    for ep in range(1, EPISODES + 1):
        epsilon = max(EPSILON_END, EPSILON_START - (ep - 1) * EPSILON_DECAY)
        state = env.reset()
        total_reward = 0
        step = 0
        done = False
        start_time = time.time()

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    logger.info("Обучение прервано пользователем")
                    if IMAGEIO_AVAILABLE and gif_frames:
                        imageio.mimsave('training.gif', gif_frames, duration=1000/FPS)
                        logger.info("GIF сохранён в training.gif")
                    pygame.quit()
                    return

            s_idx = env.states.index(state)
            a_idx = agent.select_action(s_idx, env.opened, epsilon)
            action = env.actions[a_idx]

            next_state, reward, done = env.step(action)
            next_s_idx = env.states.index(next_state)
            agent.update(s_idx, a_idx, reward, next_s_idx, done)

            log_step(step, state, action, reward)
            elapsed_time = time.time() - start_time
            draw_grid(screen, font, env, action, env.opened, done, elapsed_time)

            # Захват кадра для GIF, если imageio доступен и эпизод кратен 10
            if IMAGEIO_AVAILABLE and ep % 10 == 0:
                frame = pygame.surfarray.array3d(screen)
                frame = np.transpose(frame, (1, 0, 2))  # Переводим в формат imageio
                gif_frames.append(frame)

            clock.tick(FPS)
            total_reward += reward
            state = next_state
            step += 1

            await asyncio.sleep(1.0 / FPS)

        logger.info(f"Эпизод {ep} завершён. Общая награда: {total_reward}, EPSILON: {epsilon:.3f}")
        plotter.add(total_reward)

    np.save("q_table.npy", agent.Q)
    logger.info("Q-таблица сохранена в q_table.npy")
    if IMAGEIO_AVAILABLE and gif_frames:
        imageio.mimsave('training.gif', gif_frames, duration=1000/FPS)
        logger.info("GIF сохранён в training.gif")
    plotter.show()
    pygame.quit()
    logger.info("Обучение завершено")

async def main():
    setup()
    await train()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())