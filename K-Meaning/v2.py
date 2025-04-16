import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode

# Загрузка данных
df = pd.read_csv('/Users/Jakyp/Desktop/IAI/K-Meaning/kmeans.csv')

# Предположим, что у вас есть столбец с истинными метками (замените 'true_labels' на реальный столбец)
# Если меток нет, этот шаг нужно пропустить
if 'true_labels' not in df.columns:
    print("Столбец с истинными метками отсутствует. Accuracy не может быть вычислена.")
else:
    true_labels = df['true_labels']

# Используем все числовые признаки, кроме потенциальных ID или кластеров
X = df.select_dtypes(include='number').drop(columns=['Cluster', 'true_labels'], errors='ignore')

# Удаление выбросов (например, по 'Outstate' и 'Grad.Rate')
if 'Outstate' in X.columns and 'Grad.Rate' in X.columns:
    X = X[(X['Grad.Rate'] <= 100) & (X['Outstate'] <= 20000)]
    df = df.loc[X.index]

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Финальная модель
k_opt = 6
kmeans = KMeans(n_clusters=k_opt, random_state=42, n_init=20)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Вычисление accuracy (если есть истинные метки)
if 'true_labels' in df.columns:
    # Сопоставляем кластеры с истинными метками
    labels = df['Cluster']
    true_labels = df['true_labels']
    # Создаем словарь для соответствия кластеров и меток
    mapping = {}
    for cluster in range(k_opt):
        mask = (labels == cluster)
        # Находим наиболее частую истинную метку в этом кластере
        most_common = mode(true_labels[mask])[0][0]
        mapping[cluster] = most_common
    # Преобразуем предсказанные метки кластеров в соответствующие истинные метки
    predicted_labels = labels.map(mapping)
    # Вычисляем accuracy
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"\nAccuracy (сопоставление кластеров с истинными метками): {accuracy:.2f}")

# Вычисление центроидов
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=X.columns)
print("\nЦентроиды кластеров (в исходных единицах):")
print(centroid_df)

# Вычисление расстояний между центроидами
distances = squareform(pdist(centroids))
print("\nРасстояния между центроидами (евклидово расстояние):")
distance_df = pd.DataFrame(distances, index=[f"Cluster {i}" for i in range(k_opt)], 
                          columns=[f"Cluster {i}" for i in range(k_opt)])
print(distance_df)

# Печать кластеров и признаков
print("\nДанные с метками кластеров:")
print(df[['Cluster'] + list(X.columns)])

# Визуализация на паре признаков (пример: Outstate vs Grad.Rate)
if 'Outstate' in df.columns and 'Grad.Rate' in df.columns:
    sns.scatterplot(data=df, x='Outstate', y='Grad.Rate', hue='Cluster', palette='Set2', s=100)
    sns.scatterplot(x=centroid_df['Outstate'], y=centroid_df['Grad.Rate'], s=200, marker='X', color='black', label='Центроиды')
    plt.title('Кластеры университетов по стоимости и выпуску')
    plt.xlabel('Out-of-State Tuition')
    plt.ylabel('Graduation Rate')
    plt.grid()
    plt.show()
else:
    print("Error")