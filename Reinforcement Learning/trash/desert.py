import pygame
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Параметры среды
GRID_SIZE = 5
ACTIONS = ['↑', '↓', '←', '→']  # Вверх, Вниз, Влево, Вправо

# Цвета для Pygame
COLORS = {
    'background': (245, 245, 220),
    'agent': (0, 128, 0),
    'treasure': (255, 215, 0),
    'trap': (139, 0, 0),
    'storm': (255, 140, 0),
    'coin': (255, 255, 0),
    'grid': (150, 150, 150),
    'collected_coin': (200, 200, 200)  # Серый цвет для собранных монет
}

class DesertTreasureEnv:
    def __init__(self):
        self.grid_size = GRID_SIZE
        self.start = (0, 0)
        self.treasure = (4, 4)
        self.traps = [(2, 2), (3, 1), (4, 3)]
        self.storms = [(1, 3), (4, 0), (3, 3)]
        self.coins = [(0, 4), (2, 0), (3, 4)]
        self.collected_coins = []  # Список собранных монет
        
        # Инициализация Pygame
        pygame.init()
        self.cell_size = 80
        self.screen = pygame.display.set_mode(
            (self.grid_size * self.cell_size, self.grid_size * self.cell_size))
        pygame.display.set_caption("Сокровища в пустыне")

    def reset(self):
        self.collected_coins = []  # Сбрасываем собранные монеты
        return self.start
    
    def step(self, state, action):
        x, y = state
        done = False
        
        # Движение с 20% шансом ошибки
        if random.random() < 0.2:
            action = random.choice(ACTIONS)
        
        # Применение действия
        if action == '↑' and x > 0:
            x -= 1
        elif action == '↓' and x < self.grid_size-1:
            x += 1
        elif action == '←' and y > 0:
            y -= 1
        elif action == '→' and y < self.grid_size-1:
            y += 1
            
        next_state = (x, y)
        reward = -1  # Базовая награда за шаг
        
        # Проверка объектов в клетке
        if next_state == self.treasure:
            reward = 50
            done = True
        elif next_state in self.traps:
            reward = -20
            done = True
        elif next_state in self.storms:
            reward = -10
            next_state = (random.randint(0,4), random.randint(0,4))  # Телепортация
        elif next_state in self.coins:
            if next_state in self.collected_coins:
                reward = -5  # Штраф за повторное наступление
            else:
                reward = 10
                self.collected_coins.append(next_state)  # Добавляем монету в собранные
        
        return next_state, reward, done
    
    def render(self, state):
        self.screen.fill(COLORS['background'])
        
        # Отрисовка сетки
        for i in range(1, GRID_SIZE):
            pygame.draw.line(self.screen, COLORS['grid'], 
                           (0, i*self.cell_size), 
                           (GRID_SIZE*self.cell_size, i*self.cell_size))
            pygame.draw.line(self.screen, COLORS['grid'],
                           (i*self.cell_size, 0),
                           (i*self.cell_size, GRID_SIZE*self.cell_size))
        
        # Отрисовка объектов
        for obj in [('treasure', self.treasure),
                    *[('trap', trap) for trap in self.traps],
                    *[('storm', storm) for storm in self.storms]]:
            color = COLORS[obj[0]]
            rect = pygame.Rect(obj[1][1]*self.cell_size + 2, 
                              obj[1][0]*self.cell_size + 2,
                              self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(self.screen, color, rect)
        
        # Отрисовка монет (собранные серым цветом)
        for coin in self.coins:
            color = COLORS['collected_coin'] if coin in self.collected_coins else COLORS['coin']
            rect = pygame.Rect(coin[1]*self.cell_size + 2, 
                              coin[0]*self.cell_size + 2,
                              self.cell_size - 4, self.cell_size - 4)
            pygame.draw.rect(self.screen, color, rect)
        
        # Отрисовка агента
        pygame.draw.circle(self.screen, COLORS['agent'],
                          (state[1]*self.cell_size + self.cell_size//2,
                           state[0]*self.cell_size + self.cell_size//2),
                          self.cell_size//3)
        
        pygame.display.flip()
        pygame.time.delay(300)  # Задержка для визуализации

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        self.alpha = 0.1  # Скорость обучения
        self.gamma = 0.9  # Коэффициент дисконтирования
        self.epsilon = 0.1  # Вероятность исследования
    
    def choose_action(self, state):
        x, y = state
        if random.random() < self.epsilon:
            return random.choice(ACTIONS)
        else:
            return ACTIONS[np.argmax(self.q_table[x, y])]
    
    def update_q_table(self, state, action, reward, next_state):
        x, y = state
        next_x, next_y = next_state
        action_idx = ACTIONS.index(action)
        
        old_value = self.q_table[x, y, action_idx]
        next_max = np.max(self.q_table[next_x, next_y])
        
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[x, y, action_idx] = new_value

def train_and_visualize(episodes=500):
    env = DesertTreasureEnv()
    agent = QLearningAgent()
    
    episode_rewards = []
    moving_avg = []
    
    plt.ion()  # Интерактивный режим для matplotlib
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        step = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(state, action)
            agent.update_q_table(state, action, reward, next_state)
            
            # Вывод информации в терминал (можно уменьшить количество выводимых эпизодов)
            if episode % 50 == 0:
                print(f"Шаг {step}: Состояние {state}, Действие {action}, Награда {reward}")
                env.render(state)
            
            total_reward += reward
            state = next_state
            step += 1
        
        episode_rewards.append(total_reward)
        
        # Обновление графиков каждые 50 эпизодов
        if episode % 5 == 0:
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.plot(episode_rewards, label='Награда за эпизод')
            plt.xlabel('Эпизоды')
            plt.ylabel('Награда')
            plt.title('Динамика обучения')
            plt.tight_layout()
            plt.draw()
            plt.pause(0.1)
        
        print(f"Эпизод {episode} завершен! Общая награда: {total_reward}")
    
    plt.ioff()
    plt.show()
    pygame.quit()

if __name__ == "__main__":
    train_and_visualize(episodes=500)