import matplotlib.pyplot as plt
import pandas as pd

# Считываем Excel-таблицу в переменную data
data = pd.read_excel('weather.xls', skiprows=6)
# Удаляем пропуски
data = data[data['T'].notna()]
# Преобразуем российский формат дат для дальнейшего анализа
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)

data.head(10)

# data_train - тренировочная выборка
# data_test - тестовая выборка

data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']

plt.figure(figsize=(20, 5))
plt.plot(data_train['date'], data_train['T'], label='Train data')
plt.plot(data_test['date'], data_test['T'], label='Test data')
plt.legend()
plt.show()
