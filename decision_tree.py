# 3 модель: Decision Tree + dayofyear

import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_excel('weather.xls', skiprows=6)
data = data[data['T'].notna()]
data['date'] = pd.to_datetime(data['Местное время в Москве (ВДНХ)'], dayfirst=True)

data['dayofyear'] = data['date'].dt.dayofyear

data_train = data[data['date'] < '2020-01-01']
data_test = data[data['date'] >= '2020-01-01']
x_train = pd.DataFrame()
x_train['dayofyear'] = data_train['dayofyear']
x_test = pd.DataFrame()
x_test['dayofyear'] = data_test['dayofyear']
y_train = data_train['T']
y_test = data_test['T']

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
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

# Для 3 модели DecisionTreeRegressor + dayofyear
# Сумма ошибок на тренировочных данных = 22.840332942054822
# Сумма ошибок на тестовых данных = 35.67149875521946

# Для 4 модели KNeibour
# Сумма ошибок на тренировочных данных = 37.668456394587515
# Сумма ошибок на тестовых данных = 45.29193369632857

plt.figure(figsize=(20, 5))
plt.scatter(data_train['dayofyear'], y_train, label='Train y(x)')
plt.scatter(data_test['dayofyear'], y_test, label='Test y(x)')
plt.scatter(data_train['dayofyear'], pred_train, label='Train predict')
plt.scatter(data_test['dayofyear'], pred_test, label='Test predict')
plt.legend()
plt.show()