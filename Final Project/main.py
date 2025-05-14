import asyncio
import platform
import pygame
import time
import numpy as np
from config import EPISODES, CELL_SIZE, FPS
from environment import MinesweeperEnv
from q_learning import QLearningAgent
from utils import Plotter, draw_grid

def log_step(step, state, action, reward):
    """Log the details of each step for debugging."""
    print(f"Шаг {step}: Состояние={state}, Действие={action}, Награда={reward}")

env = None
agent = None
screen = None
font = None
clock = None
plotter = None

def setup():
    """Initialize the game environment and Pygame."""
    global env, agent, screen, font, clock, plotter
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    plotter = Plotter()

    pygame.init()
    screen = pygame.display.set_mode((env.cols * CELL_SIZE, env.rows * CELL_SIZE))
    pygame.display.set_caption("Minesweeper Q-Learning")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 24)

async def update_loop():
    """Run the Q-learning training loop with visualization."""
    global env, agent, screen, font, clock, plotter
    
    EPSILON_START = 1.0
    EPSILON_END = 0.0001
    EPSILON_DECAY = 0.004

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
            clock.tick(FPS)

            total_reward += reward
            state = next_state
            step += 1

            await asyncio.sleep(1.0 / FPS)  # Control frame rate

        print(f"Эпизод {ep} завершён с общей наградой {total_reward}, EPSILON={epsilon:.3f} \n")
        plotter.add(total_reward)

    # Save the trained Q-table
    np.save("q_table.npy", agent.Q)
    plotter.show()

async def main():
    """Main async function to run the game."""
    setup()
    await update_loop()

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())