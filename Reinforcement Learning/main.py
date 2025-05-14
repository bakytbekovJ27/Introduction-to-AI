import pygame
from config import EPISODES, CELL_SIZE, FPS
from environment import MinesweeperEnv
from q_learning import QLearningAgent
from utils import Plotter

def log_step(step, state, action, reward):
    print(f"Шаг {step}: Состояние={state}, Действие={action}, Награда={reward}")

def main():
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    plotter = Plotter()

    # pygame.init()
    # screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE))
    # pygame.display.set_caption("Minesweeper Q-Learning")
    # clock = pygame.time.Clock()
    # font = pygame.font.SysFont("arial", 24)

    EPSILON_START = 1.0
    EPSILON_END = 0.0001
    EPSILON_DECAY = 0.004

    for ep in range(1, EPISODES + 1):
        epsilon = max(EPSILON_END, EPSILON_START - (ep - 1) * EPSILON_DECAY)
        state = env.reset()
        total = 0
        step = 0
        done = False

        while not done:
            # # for event in pygame.event.get():
            #     if event.type == pygame.QUIT:
            #         pygame.quit()
            #         return

            s_idx = env.states.index(state)
            a_idx = agent.select_action(s_idx, env.opened, epsilon)
            action = env.actions[a_idx]

            next_state, reward, done = env.step(action)
            next_s_idx = env.states.index(next_state)
            agent.update(s_idx, a_idx, reward, next_s_idx, done)

            log_step(step, state, action, reward)
            # draw_grid(screen, font, env, action, env.opened, done)
            # clock.tick(FPS)

            total += reward
            state = next_state
            step += 1

        print(f"Эпизод {ep} завершён с общей наградой {total}, EPSILON={epsilon:.3f}\n")
        plotter.add(total)

    plotter.show()
    # pygame.quit()

if __name__ == '__main__':
    main()