import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Загружаем и подготавливаем данные
data = pd.read_excel('weather.xls', skiprows=6)
data = data[data['T'].notna()]
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)

# Создаем новый признак - день в году
data['dayofyear'] = data['date'].dt.dayofyear

# Рассчитываем корреляцию между числовыми признаками
correlation_matrix = data.corr()

# Визуализируем матрицу корреляции
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix')
plt.show()
