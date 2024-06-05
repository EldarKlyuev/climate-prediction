import matplotlib.pyplot as plt

# Подключаем библиотеку pandas для работы с таблицами с псевдонимом pd
import pandas as pd

# Считываем Excel-таблицу в переменную data
data = pd.read_excel('weather.xls')

# Смотрим, что получилось
# первые 10 строк
print(data.head(10))

# Удаляем лишние комментарии (первые 6 строк)
data = pd.read_excel('weather.xls', skiprows=6)
print(data.head(10))

# pandas.DataFrame - сложный тип данных, объект таблицы
print(data.shape)

# Посмотрим на колонки, которые нам доступны
print(data.columns)

# Столбец - отдельный объект
print(data['T'])
print(data['T'][4])
print(data['Местное время в Москве (ВДНХ)'][4])

# Нарисуем график температуры от времени
# Для этого нужно преобразовать колонку с датами в специальный формат дат
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)

# Распечатаем график температуры
x = data['date']
y = data['T']

plt.figure(figsize=(20, 5))
plt.plot(x, y)
plt.show()

# Есть ли в данных пропуски?
condition = data['T'].isna()
print(data[condition])

# Удаляем некорректные данные
good_condition = data['T'].notna()
data = data[good_condition]

# Максимум, минимум, среднее
data['T'].min(), data['T'].max(), data['T'].mean()

# Гистограмма
data['T'].hist()

# Выбираем диапазон данных для отображения
condition = data['date'] < '2023-01-01'
short_data = data[condition]

x = short_data['date']
y = short_data['T']

plt.figure(figsize=(20, 5))
plt.plot(x, y)
plt.show()