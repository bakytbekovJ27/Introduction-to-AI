# Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Загрузка данных (train.csv и test.csv)
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Удаляем строки, где SalePrice отсутствует (на всякий случай)
train_data = train_data.dropna(subset=['SalePrice'])

# Разделение на признаки и целевую переменную в train.csv
X_train_full = train_data.drop('SalePrice', axis=1)
y_train_full = train_data['SalePrice']

# Разделение данных (70% train, 30% test)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.3, random_state=42)

# Числовые и категориальные признаки
numerical_cols = X_train_full.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train_full.select_dtypes(include=['object']).columns

# Обработка числовых данных (заполнение пропусков + масштабирование)
num_imputer = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_num = num_imputer.fit_transform(X_train[numerical_cols])
X_val_num = num_imputer.transform(X_val[numerical_cols])
X_test_num = num_imputer.transform(test_data[numerical_cols])

X_train_num = scaler.fit_transform(X_train_num)
X_val_num = scaler.transform(X_val_num)
X_test_num = scaler.transform(X_test_num)

# Обработка категориальных данных (заполнение + OneHot)
cat_imputer = SimpleImputer(strategy='most_frequent')
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
X_val_cat = cat_imputer.transform(X_val[categorical_cols])
X_test_cat = cat_imputer.transform(test_data[categorical_cols])

X_train_cat = encoder.fit_transform(X_train_cat)
X_val_cat = encoder.transform(X_val_cat)
X_test_cat = encoder.transform(X_test_cat)

# Объединение числовых и категориальных признаков
X_train_processed = np.hstack([X_train_num, X_train_cat])
X_val_processed = np.hstack([X_val_num, X_val_cat])
X_test_processed = np.hstack([X_test_num, X_test_cat])

# Обучение модели на тренировочных данных
model = LinearRegression()
model.fit(X_train_processed, y_train)

# Предсказание на валидационных данных
y_val_pred = model.predict(X_val_processed)

# Метрики на валидационных данных
r2 = r2_score(y_val, y_val_pred) * 100  # Процент точности
rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))

# Предсказание на тестовых данных
test_data_predictions = model.predict(X_test_processed)

# Сохранение предсказаний в CSV файл
test_data['SalePrice'] = test_data_predictions
test_data[['Id', 'SalePrice']].to_csv('test_predictions.csv', index=False)
print("Предсказания сохранены в файл test_predictions.csv")

print(f'Точность модели на валидационных данных: {r2:.2f}%')  
print(f'RMSE на валидационных данных: {rmse:.2f}')


# График: жилая площадь vs. предсказанная цена на валидационных данных
plt.figure(figsize=(12, 8))
plt.scatter(X_val['GrLivArea'], y_val, alpha=0.6, c='blue', label='Actual')  # Фактические значения
plt.scatter(X_val['GrLivArea'], y_val_pred, alpha=0.6, c='green', label='Predicted')  # Предсказанные значения
plt.title(f'Зависимость цены от квадратуры дома')
plt.xlabel('Жилая площадь (кв.футы)')  
plt.ylabel('Цена ($)')   
plt.grid(True)
plt.legend()

# Линия тренда (для предсказанных значений)
z = np.polyfit(X_val['GrLivArea'], y_val_pred, 1)  # Обучение на площади → цене
p = np.poly1d(z)
plt.plot(X_val['GrLivArea'], p(X_val['GrLivArea']), "r--", label='Trend Line')  # Красная линия

plt.show()