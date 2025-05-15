import numpy as np
import random  
from config import MAP

class MinesweeperEnv:
    """
    Среда «Сапёр» для агента с Q-обучением.
    Управляет состоянием игры: генерация карты, действия, награды.
    """
    def __init__(self):
        """
        Инициализация среды с картой из config.py.
        Задаёт состояния, действия, определяет мины и безопасные клетки.
        """
        self.rows, self.cols = len(MAP), len(MAP[0])  # Размеры поля
        self.states = [(i, j) for i in range(self.rows) for j in range(self.cols)]  # Все возможные клетки
        self.actions = self.states.copy()  # Действия — открыть любую клетку
        self._generate_map()  # Генерация мин на поле
        self.safe_cells = [s for s in self.states if s not in self.mines]  # Безопасные клетки

    def _generate_map(self):
        """
        Генерирует игровое поле на основе MAP из config.py.
        Определяет расположение мин (обозначены 'X').
        """
        self.mines = []
        for i, row in enumerate(MAP):
            for j, val in enumerate(row):
                if val == 'X':
                    self.mines.append((i, j))
        self.total_safe = len(self.states) - len(self.mines)  # Количество безопасных клеток

    def reset(self):
        """
        Сброс состояния игры для нового эпизода.
        Возвращает случайную безопасную клетку как начальное состояние.
        """
        self.opened = set()      # Открытые клетки
        self.game_over = False   # Флаг окончания игры
        return random.choice(self.safe_cells)

    def step(self, action):
        """
        Выполняет действие (открывает клетку) и возвращает следующее состояние, награду и флаг завершения.
        - Попадание на мину: награда -10, игра окончена.
        - Клетка уже открыта: награда -5.
        - Безопасная клетка: награда 15, +100 если открыты все безопасные клетки.
        """
        if action in self.mines:
            reward = -10
            self.game_over = True
        elif action in self.opened:
            reward = -5
        else:
            self.opened.add(action)
            mine_count = self.get_mine_count(action)
            reward = 15
            if len(self.opened) == self.total_safe:
                reward += 100
                self.game_over = True
        return action, reward, self.game_over

    def get_mine_count(self, pos):
        """
        Считает количество мин вокруг заданной клетки.
        Проверяет все 8 соседних клеток.
        """
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