import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('weather.xls', skiprows=6)
data = data[data['T'].notna()]
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)

data['dayofyear'] = data['date'].dt.dayofyear
import numpy as np
# Переходим в радианы, создаем новую переменную с cos
data['cos_dayofyear'] = np.cos((data['dayofyear'] - 1) / 366 * 2 * np.pi)

data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']
x_train = pd.DataFrame()
x_train['cos_dayofyear'] = data_train['cos_dayofyear']
x_test = pd.DataFrame()
x_test['cos_dayofyear'] = data_test['cos_dayofyear']
y_train = data_train['T']
y_test = data_test['T']

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train, y_train)

# Прогноз для тренировочных данных
pred_train = model.predict(x_train)
# Прогноз для данных, которые модель еще не видела
pred_test = model.predict(x_test)

from sklearn.metrics import mean_squared_error
print('Сумма ошибок на тренировочных данных =', mean_squared_error(y_train, pred_train))
print('Сумма ошибок на тестовых данных =', mean_squared_error(y_test, pred_test))

# Для 1 модели LinearRegression + dayofyear
# Сумма ошибок на тренировочных данных = 106.65218015974331
# Сумма ошибок на тестовых данных = 108.35324573271384

# Для 2 модели LinearRegression + cos_dayofyear
# Сумма ошибок на тренировочных данных = 30.03398645088782
# Сумма ошибок на тестовых данных = 32.34520984726374

from sklearn.metrics import r2_score

print('R^2 на тренировочных данных =', r2_score(y_train, pred_train))
print('R^2 на тестовых данных =', r2_score(y_test, pred_test))

plt.figure(figsize=(20, 5))
plt.scatter(data_train['dayofyear'], y_train, label='Train y(x)')
plt.scatter(data_test['dayofyear'], y_test, label='Test y(x)')
plt.scatter(data_test['dayofyear'], pred_test, label='Test predict', color='red')
plt.legend()
plt.show()