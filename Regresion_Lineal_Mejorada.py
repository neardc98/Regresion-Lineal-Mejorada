import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.indexes.datetimes import date_range
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('./Datos/salarios1.csv')
print(dataset.head())
print(dataset.columns)
print(dataset.shape)

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, Y_train, Y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

# crear el modelo.
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

vis_train = plt
vis_train.scatter(X_train, Y_train, color='blue')
vis_train.plot(X_train, regressor.predict(X_train), color='black')
vis_train.title('Salario Vs Experiencia')
vis_train.xlabel('Experiencia')
vis_train.ylabel('Salarios')
vis_train.show()

vis_train = plt
vis_train.scatter(X_test, Y_test, color='blue')
vis_train.plot(X_test, regressor.predict(X_test), color='black')
vis_train.title('Salario Vs Experiencia')
vis_train.xlabel('Experiencia')
vis_train.ylabel('Salarios')
vis_train.show()

print("Modelo tiene un porcentarce de score: ", regressor.score(X_test, Y_test))
