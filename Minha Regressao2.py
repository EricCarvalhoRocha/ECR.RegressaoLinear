#importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#adquirindo os dados
dataset = pd.read_csv('fatal-accidents-per-million-flights.csv')
x = dataset.iloc[:,2].values
y = dataset.iloc[:, -1].values

#separando os dados em treino e teste
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#transformando em uma matriz coluna
x_train = x_train.reshape(-1, 1) 
x_test = x_test.reshape(-1, 1)

#preparando o modelo
from sklearn.linear_model import LinearRegression
modelo = LinearRegression() 
modelo.fit(x_train, y_train)

y_pred = modelo.predict(x_test)

#visualizando os dado juntamente com a regressao linear
plt.scatter(x_train,y_train, color = 'red')
plt.plot(x_train, modelo.predict(x_train), color='blue')
plt.title('Acidentes fatais por milhão de voos')
plt.xlabel('Ano')
plt.ylabel('Acidentes fatais')
plt.show()

#testando o modelo juntamente com os dados de teste
plt.scatter(x_test,y_test, color = 'red')
plt.plot(x_train, modelo.predict(x_train), color='blue')
plt.title('Acidentes fatais por milhão de voos')
plt.xlabel('Ano')
plt.ylabel('Acidentes fatais')
plt.show()

print(modelo.predict([[2017]]))
print(modelo.coef_)
print(modelo.intercept_)

#calculando o erro quadratico
y_pred = modelo.predict(x_test)
def soma_dos_erros_quadraticos():
    erro = y_test - y_pred
    return sum(erro)**2

soma_dos_erros_quadraticos()    


