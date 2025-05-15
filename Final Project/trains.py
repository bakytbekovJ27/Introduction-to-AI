import asyncio
import platform
import pygame
import time
import numpy as np
import imageio
import os
import random
from config import MAP, ALPHA, GAMMA, EPISODES, CELL_SIZE, FPS, SAVE_GIF

class MinesweeperEnv:
    def __init__(self):
        """Initialize environment using MAP from config"""
        self.rows, self.cols = len(MAP), len(MAP[0])
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]
        self.actions = self.states.copy()
        self._generate_map()
        self.total_safe = len(self.states) - len(self.mines)
        self.reset()

    def _generate_map(self):
        """Generate mines based on MAP from config"""
        self.mines = []
        for i, row in enumerate(MAP):
            for j, val in enumerate(row):
                if val == 'X':
                    self.mines.append((i, j))

    def reset(self):
        """Reset environment for new episode"""
        self.opened = set()
        self.game_over = False
        # Start from random safe cell
        initial_state = random.choice([s for s in self.states if s not in self.mines])
        self.opened.add(initial_state)
        return initial_state

    def step(self, action):
        """Execute action and return (next_state, reward, done)"""
        reward = -1  # Default step penalty
        done = False

        if action in self.mines:
            reward = -20  # Mine penalty
            done = True
        elif action in self.opened:
            reward = -1  # Already opened penalty
        else:
            self.opened.add(action)
            mine_count = self._count_adjacent_mines(action)
            
            if mine_count > 0:
                reward = 15  # Reward for finding number
            else:
                reward = 5  # Reward for empty cell
            
            if len(self.opened) == self.total_safe:
                reward += 100  # Win bonus
                done = True

        return action, reward, done

    def _count_adjacent_mines(self, pos):
        """Count mines around given position"""
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

class QLearningAgent:
    def __init__(self, states, actions):
        """Initialize Q-learning agent with parameters from config"""
        self.states = states
        self.actions = actions
        self.Q = np.zeros((len(states), len(actions)))
        self.alpha = ALPHA  # Learning rate from config
        self.gamma = GAMMA  # Discount factor from config

    def select_action(self, state_idx, opened, epsilon):
        """Epsilon-greedy action selection"""
        available_actions = [i for i, a in enumerate(self.actions) 
                           if a not in opened]
        
        if not available_actions:
            return random.choice(range(len(self.actions)))
            
        if random.random() < epsilon:
            return random.choice(available_actions)
        else:
            q_values = [self.Q[state_idx, a] for a in available_actions]
            return available_actions[np.argmax(q_values)]

    def update(self, state_idx, action_idx, reward, next_state_idx, done):
        """Update Q-table using Q-learning algorithm"""
        current_q = self.Q[state_idx, action_idx]
        max_next_q = np.max(self.Q[next_state_idx]) if not done else 0
        target = reward + self.gamma * max_next_q
        self.Q[state_idx, action_idx] += self.alpha * (target - current_q)

    def save_model(self, filename):
        """Save Q-table to file"""
        os.makedirs("utils", exist_ok=True)
        np.save(filename, self.Q)

class TrainingVisualizer:
    """Handles training visualization and GIF creation"""
    def __init__(self, env, cell_size=CELL_SIZE):
        pygame.init()
        self.env = env
        self.cell_size = cell_size
        self.screen = pygame.display.set_mode(
            (env.cols * cell_size, env.rows * cell_size))
        pygame.display.set_caption("Minesweeper Q-Learning Training")
        self.font = pygame.font.SysFont("Arial", 24)
        self.clock = pygame.time.Clock()
        self.frames = []

    def render(self, current_action, episode, epsilon, reward):
        """Render current game state"""
        self.screen.fill((255, 255, 255))  # White background
        
        # Draw grid
        for i in range(self.env.rows):
            for j in range(self.env.cols):
                pos = (i, j)
                rect = pygame.Rect(j * self.cell_size, 
                                 i * self.cell_size, 
                                 self.cell_size, 
                                 self.cell_size)
                
                # Closed cell
                if pos not in self.env.opened:
                    pygame.draw.rect(self.screen, (192, 192, 192), rect)
                # Opened cell
                else:
                    pygame.draw.rect(self.screen, (255, 255, 255), rect)
                    mine_count = self.env._count_adjacent_mines(pos)
                    if mine_count > 0:
                        text = self.font.render(str(mine_count), True, (0, 0, 0))
                        self.screen.blit(text, 
                                        (j * self.cell_size + 30, 
                                         i * self.cell_size + 25))
                
                # Mine (visible when game over)
                if self.env.game_over and pos in self.env.mines:
                    pygame.draw.rect(self.screen, (255, 0, 0), rect)
                
                # Current action highlight
                if pos == current_action:
                    pygame.draw.rect(self.screen, (0, 0, 255), rect, 3)
                
                pygame.draw.rect(self.screen, (0, 0, 0), rect, 1)  # Border

        # Display episode info
        info_text = f"Episode: {episode}  Epsilon: {epsilon:.2f}  Reward: {reward}"
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        
        # Capture frame for GIF
        if SAVE_GIF and episode % 10 == 0:  # Sample every 10 episodes
            self._capture_frame()

    def _capture_frame(self):
        """Capture current frame for GIF"""
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(np.flipud(frame))

    def save_gif(self, filename):
        """Save collected frames as GIF"""
        if self.frames:
            imageio.mimsave(filename, self.frames, fps=FPS//2)  # Half speed for better viewing

    def close(self):
        """Clean up resources"""
        pygame.quit()

class TrainingStats:
    """Tracks and visualizes training statistics"""
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_values = []

    def add_episode(self, reward, length, epsilon):
        """Record episode statistics"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)

    def plot_progress(self, window_size=100):
        """Plot training progress with moving average"""
        try:
            import matplotlib.pyplot as plt
            
            # Calculate moving averages
            def moving_average(data, window):
                return np.convolve(data, np.ones(window)/window, mode='valid')
            
            rewards_ma = moving_average(self.episode_rewards, window_size)
            lengths_ma = moving_average(self.episode_lengths, window_size)
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Reward plot
            plt.subplot(1, 2, 1)
            plt.plot(rewards_ma)
            plt.title(f'Rewards (MA {window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            
            # Length plot
            plt.subplot(1, 2, 2)
            plt.plot(lengths_ma)
            plt.title(f'Episode Length (MA {window_size})')
            plt.xlabel('Episode')
            plt.ylabel('Average Length')
            
            plt.tight_layout()
            os.makedirs("utils", exist_ok=True)
            plt.savefig('utils/training_progress.png')
            plt.close()
            
        except ImportError:
            print("Matplotlib not available - skipping progress plot")

async def train_minesweeper_agent():
    """Main training function"""
    # Initialize components
    env = MinesweeperEnv()
    agent = QLearningAgent(env.states, env.actions)
    visualizer = TrainingVisualizer(env)
    stats = TrainingStats()
    
    # Training parameters
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.0005
    save_interval = 500  # Save model every N episodes
    
    try:
        # Training loop
        for episode in range(1, EPISODES + 1):
            # Calculate epsilon (exponential decay)
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                     np.exp(-epsilon_decay * episode)
            
            # Initialize episode
            state = env.reset()
            total_reward = 0
            step = 0
            done = False
            
            while not done:
                # Handle pygame events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise KeyboardInterrupt
                
                # Select and execute action
                state_idx = env.states.index(state)
                action_idx = agent.select_action(state_idx, env.opened, epsilon)
                action = env.actions[action_idx]
                
                next_state, reward, done = env.step(action)
                next_state_idx = env.states.index(next_state)
                
                # Update Q-table
                agent.update(state_idx, action_idx, reward, next_state_idx, done)
                
                # Render and record
                visualizer.render(action, episode, epsilon, total_reward)
                total_reward += reward
                state = next_state
                step += 1
                
                # Control frame rate
                await asyncio.sleep(1.0 / FPS)
            
            # Record episode stats
            stats.add_episode(total_reward, step, epsilon)
            
            # Print progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{EPISODES} - "
                      f"Reward: {total_reward} - "
                      f"Epsilon: {epsilon:.3f} - "
                      f"Steps: {step}")
            
            # Save intermediate model
            if episode % save_interval == 0:
                agent.save_model(f"utils/model_episode_{episode}.npy")
        
        # Training complete
        print("Training finished!")
        
        # Save final results
        agent.save_model("utils/final_model.npy")
        visualizer.save_gif("utils/training_progress.gif")
        stats.plot_progress()
        
    except KeyboardInterrupt:
        print("Training interrupted - saving progress...")
        agent.save_model("utils/interrupted_model.npy")
    
    finally:
        visualizer.close()

if __name__ == "__main__":
    if platform.system() == "Emscripten":
        asyncio.ensure_future(train_minesweeper_agent())
    else:
        asyncio.run(train_minesweeper_agent())