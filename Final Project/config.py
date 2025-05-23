# Конфигурационные настройки для проекта "Сапёр" с Q-обучением
# Определяет игровое поле и гиперпараметры

 # Игровое поле: точки — пустые клетки, X — мины
MAP = [
    ".........",  
    "..X......",
    "....X....",
    ".X....X..",
    "......X..",
    "X........",
    "...X.....",
    ".....X...",
    ".X......X"
]

# Параметры Q-обучения
ALPHA = 0.1      # Скорость обучения (learning rate)
GAMMA = 0.9      # Коэффициент дисконтирования (discount factor)
EPISODES = 5000 # Количество обучающих эпизодов
CELL_SIZE = 80   # Размер клетки в пикселях для Pygame
FPS = 10         # Кадров в секунду для визуализации в Pygame

# Прочие настройки
SAVE_PLOTS = True         # Сохранять ли графики обучения
SAVE_GIF = True  # Доступна ли библиотека imageio для сохранения анимаций
RICH_AVAILABLE = True
