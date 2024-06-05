import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Загружаем и подготавливаем данные
data = pd.read_excel('weather.xls', skiprows=6)
data = data[data['T'].notna()]
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)

# Создаем новый признак - день в году
data['dayofyear'] = data['date'].dt.dayofyear

# Разделяем данные на тренировочную и тестовую выборки
data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']

x_train = pd.DataFrame()
x_train['dayofyear'] = data_train['dayofyear']
x_test = pd.DataFrame()
x_test['dayofyear'] = data_test['dayofyear']

y_train = data_train['T']
y_test = data_test['T']

# Обучаем модель случайного леса
n_estimators = 1000  # Количество деревьев в лесу
model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
model.fit(x_train, y_train)

# Прогноз для тренировочных данных
pred_train = model.predict(x_train)
# Прогноз для данных, которые модель еще не видела
pred_test = model.predict(x_test)

# Проверяем качество численно
print('Сумма ошибок на тренировочных данных =', mean_squared_error(y_train, pred_train))
print('Сумма ошибок на тестовых данных =', mean_squared_error(y_test, pred_test))

from sklearn.metrics import r2_score

print('R^2 на тренировочных данных =', r2_score(y_train, pred_train))
print('R^2 на тестовых данных =', r2_score(y_test, pred_test))

# Визуализируем результаты
plt.figure(figsize=(20, 5))
plt.scatter(x_train['dayofyear'], y_train, label='Train y(x)')
plt.scatter(x_test['dayofyear'], y_test, label='Test y(x)')
plt.scatter(x_test['dayofyear'], pred_test, label='Test predict', color='red')
plt.legend()
plt.show()
