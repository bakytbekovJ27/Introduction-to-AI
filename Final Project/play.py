import pygame
import numpy as np
from config import CELL_SIZE, FPS, MAP
import random
import time

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

    def select_action(self, state_idx, opened, epsilon=0):  # Epsilon=0 для использования обученной модели
        available_actions = [i for i, action in enumerate(self.actions) if action not in opened]
        if not available_actions:
            return np.random.randint(len(self.actions))
        q_values = [self.Q[state_idx, a] for a in available_actions]
        return available_actions[np.argmax(q_values)]

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

def play_human_vs_model():
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    try:
        agent.Q = np.load("q_table.npy")
    except FileNotFoundError:
        return

    pygame.init()
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE))
    pygame.display.set_caption("Сапёр: Человек против бота")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

    state = env.reset()  # Открывает безопасную клетку автоматически
    human_turn = False  # Бот ходит первым после открытия безопасной клетки
    game_over = False
    start_time = time.time()
    last_player = None  # Отслеживаем, кто сделал последний ход

    # Отрисовка начального состояния, чтобы избежать черного экрана
    draw_grid(screen, font, env, state, env.opened, game_over, 0)

    while not game_over:
        if human_turn:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    pos = pygame.mouse.get_pos()
                    col = pos[0] // CELL_SIZE
                    row = pos[1] // CELL_SIZE
                    action = (row, col)
                    if action not in env.opened and action in env.states:
                        next_state, reward, game_over = env.step(action)
                        state = next_state
                        last_player = "человек"
                        elapsed_time = time.time() - start_time
                        draw_grid(screen, font, env, action, env.opened, game_over, elapsed_time)
                        if not game_over:
                            human_turn = False
        else:
            s_idx = env.states.index(state)
            a_idx = agent.select_action(s_idx, env.opened)
            action = env.actions[a_idx]
            next_state, reward, game_over = env.step(action)
            state = next_state
            last_player = "агент"
            elapsed_time = time.time() - start_time
            draw_grid(screen, font, env, action, env.opened, game_over, elapsed_time)
            if not game_over:
                human_turn = True

        if game_over:
            if len(env.opened) == env.total_safe:
                winner = last_player
                print(f"Победа! Выиграл {winner.capitalize()}!")
            else:
                loser = last_player
                winner = "человек" if loser == "агент" else "агент"
                print(f"Игра окончена. Проиграл {loser.capitalize()}. Выиграл {winner.capitalize()}!")

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    play_human_vs_model()