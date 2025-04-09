import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib  # Для сохранения модели

# Установка опции для избежания предупреждений
pd.set_option('future.no_silent_downcasting', True)

# Загрузка данных
data = pd.read_csv('titanic.csv')

# Подготовка признаков (X)
X = data[['Pclass', 'Sex', 'Age', 'Cabin', 'Fare', 'Embarked']].copy()

# Преобразование пола в числа
X['Sex'] = X['Sex'].replace({'male': 0, 'female': 1})

# Заполнение пропусков в возрасте средним значением
X['Age'] = X['Age'].fillna(data['Age'].mean())

# Обработка Cabin: извлекаем первую букву или "U" для пропусков
X['Cabin'] = X['Cabin'].fillna('U').apply(lambda x: x[0])
X = pd.concat([X.drop('Cabin', axis=1), pd.get_dummies(X['Cabin'], prefix='Cabin')], axis=1)

# Обработка Embarked: кодируем порт посадки (C, Q, S) в dummy-переменные
X['Embarked'] = X['Embarked'].fillna('S')
X = pd.concat([X.drop('Embarked', axis=1), pd.get_dummies(X['Embarked'], prefix='Embarked')], axis=1)

# Целевая переменная
y = data['Survived']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Оценка точности
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.2f}")

# Примеры предсказаний
print("Примеры предсказаний:")
for i in range(5):
    print(f"Реальный: {y_test.iloc[i]}, Предсказанный: {y_pred[i]}")

# Сохраняем модель и список признаков
joblib.dump(model, 'titanic_model.pkl')
joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')
print("Модель и структура признаков сохранены.")
