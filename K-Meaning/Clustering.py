import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Загрузка данных
df = pd.read_csv('/Users/Jakyp/Desktop/IAI/K-Meaning/kmeans.csv')

# Используем все числовые признаки, кроме потенциальных ID или кластеров
X = df.select_dtypes(include='number').drop(columns=['Cluster'], errors='ignore')

# Масштабирование
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Финальная модель
k_opt = 5  # Фиксированное число кластеров
kmeans = KMeans(n_clusters=k_opt, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Печать кластеров и признаков
print(df[['Cluster'] + list(X.columns)])

# Визуализация на паре признаков (пример: Outstate vs Grad.Rate)
if 'Outstate' in df.columns :
    sns.scatterplot(data=df, x='Outstate', y='Grad.Rate', hue='Cluster', palette='Set2', s=100)
    plt.title('Кластеры университетов по стоимости и выпуску')
    plt.xlabel('Out-of-State Tuition')
    plt.ylabel('Graduation Rate')
    plt.grid()
    plt.show()
else:
    print("Error")