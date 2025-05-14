import numpy as np
from config import ALPHA, GAMMA

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