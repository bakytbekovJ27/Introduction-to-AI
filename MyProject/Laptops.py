import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка данных
data = pd.read_csv('laptop_price.csv')

# Функция для извлечения частоты процессора
def extract_cpu_frequency(cpu_str):
    match = re.search(r'(\d+\.\d+)', str(cpu_str))
    return float(match.group(1)) if match else np.nan

data['Cpu_Freq_GHz'] = data['Cpu'].apply(extract_cpu_frequency)

# Функция для извлечения оперативной памяти (RAM)
def extract_ram(ram_str):
    match = re.search(r'(\d+)', str(ram_str))
    return int(match.group(1)) if match else np.nan

data['RAM_GB'] = data['Ram'].apply(extract_ram)

# Определяем, есть ли SSD (1 = SSD, 0 = HDD)
data['SSD'] = data['Memory'].apply(lambda x: 1 if 'SSD' in str(x) else 0)

# Кодируем бренд ноутбука (категориальный признак)
data['Brand'] = data['Company'].astype('category').cat.codes

# Убираем строки с пропущенными значениями
data = data.dropna()

# Выбираем признаки для модели
X = data[['Cpu_Freq_GHz', 'RAM_GB', 'SSD', 'Brand']]
y = data['Price_euros']

# Разделяем данные на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создаём и обучаем модель (случайный лес)
model = RandomForestRegressor(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Прогнозируем цены ноутбуков
y_pred = model.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("📌 Mean Squared Error:", mse)
print("📈 R2 Score:", r2)

# Визуализация реальных и предсказанных значений
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, label="Предсказания")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2, label="Идеальное предсказание")
plt.xlabel("Фактическая цена (€)")
plt.ylabel("Предсказанная цена (€)")
plt.title("Сравнение реальных и предсказанных цен ноутбуков")
plt.legend()
plt.show()