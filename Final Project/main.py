import asyncio
import platform
import pygame
import time
import numpy as np
from config import EPISODES, CELL_SIZE, FPS
from environment import MinesweeperEnv
from q_learning import QLearningAgent
from utils import Plotter, draw_grid


class MinesweeperGame:
    def __init__(self):
        self.env = None
        self.agent = None
        self.screen = None
        self.font = None
        self.clock = None
        self.plotter = None

    def setup(self):
        """Initialize the game environment and Pygame."""
        self.env = MinesweeperEnv()
        self.agent = QLearningAgent(self.env.states, self.env.actions)
        self.plotter = Plotter()

        pygame.init()
        self.screen = pygame.display.set_mode((self.env.cols * CELL_SIZE, 
                                             self.env.rows * CELL_SIZE))
        pygame.display.set_caption("Minesweeper Q-Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 24)

    def log_step(self, step, state, action, reward):
        """Log the details of each step for debugging."""
        print(f"Шаг {step}: Состояние={state}, Действие={action}, Награда={reward}")

    async def update_loop(self):
        """Run the Q-learning training loop with visualization."""
        EPSILON_START = 1.0
        EPSILON_END = 0.0001
        EPSILON_DECAY = 0.001

        for episode in range(1, EPISODES + 1):
            # Calculate epsilon for current episode
            epsilon = max(EPSILON_END, EPSILON_START - (episode - 1) * EPSILON_DECAY)
            
            # Initialize episode variables
            state = self.env.reset()
            total_reward = 0
            step = 0
            done = False
            start_time = time.time()

            while not done:
                # Handle Pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Get current state index and select action
                state_idx = self.env.states.index(state)
                action_idx = self.agent.select_action(state_idx, self.env.opened, epsilon)
                action = self.env.actions[action_idx]

                # Perform action and get next state
                next_state, reward, done = self.env.step(action)
                next_state_idx = self.env.states.index(next_state)
                
                # Update Q-table
                self.agent.update(state_idx, action_idx, reward, next_state_idx, done)

                # Logging and visualization
                self.log_step(step, state, action, reward)
                elapsed_time = time.time() - start_time
                draw_grid(self.screen, self.font, self.env, action, 
                         self.env.opened, done, elapsed_time)
                self.clock.tick(FPS)

                # Update episode progress
                total_reward += reward
                state = next_state
                step += 1

                await asyncio.sleep(1.0 / FPS)  # Control frame rate

            # Episode summary
            print(f"Эпизод {episode} завершён с общей наградой {total_reward}, "
                  f"EPSILON={epsilon:.3f}\n")
            self.plotter.add(total_reward)

        # Show training results plot
        self.plotter.show()

async def main():
    """Main async function to run the game."""
    game = MinesweeperGame()
    game.setup()
    await game.update_loop()

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(main())
    else:
        asyncio.run(main())