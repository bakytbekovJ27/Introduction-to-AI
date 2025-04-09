import pandas as pd
import joblib

# Загрузка модели и структуры признаков
model = joblib.load('titanic_model.pkl')
model_columns = joblib.load('model_columns.pkl')

new_data = pd.DataFrame({
    'Pclass': [3],
    'Sex': [0],
    'Age': [25],
    'Fare': [7.25],
    'Cabin_U': [1],
    'Cabin_C': [0],
    'Cabin_B': [0],
    'Cabin_E': [0],
    'Cabin_D': [0],
    'Cabin_A': [0],
    'Cabin_F': [0],
    'Cabin_G': [0],
    'Embarked_S': [1],
    'Embarked_C': [0],
    'Embarked_Q': [0]
})

# Добавляем недостающие признаки
for col in model_columns:
    if col not in new_data.columns:
        new_data[col] = 0

# Приводим колонки в нужный порядок
new_data = new_data[model_columns]

# Предсказание
prediction = model.predict(new_data)
for i in range(10):
    print(f"Предсказание: {prediction[0]} (1 - выжил, 0 - нет)")
