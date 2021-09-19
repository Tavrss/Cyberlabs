#Imports
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
%matplotlib inline
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
from tensorflow.keras import layers, models

#-------------------------------------- TRATAMENTO DOS DADOS  --------------------------------------

#~~~~~~~~~~~~ Importando os Dados ~~~~~~~~~~~~

from keras.datasets import mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Para esse desafio queremos apenas os dados dos dígitos 0 e 5

#~~~~~~~~~~~~ Dividindo os Dados ~~~~~~~~~~~~

X_0_train = X_train[y_train==0]
y_0_train = np.zeros(len(X_0_train))
X_0_test = X_test[y_test==0]
y_0_test = np.zeros(len(X_0_test))
X_5_train = X_train[y_train==5]
y_5_train = np.ones(len(X_5_train))*5
X_5_test = X_test[y_test==5]
y_5_test = np.ones(len(X_5_test))*5

#~~~~~~~~~~~~ Concatenando e embaralhando ~~~~~~~~~~~~

X_train  = np.append(X_5_train,X_0_train,0)
indexes1 = np.arange(0,len(X_train))
np.random.shuffle(indexes1)
X_train = X_train[[indexes1]]

y_train = np.append(y_5_train,y_0_train,0)
y_train = y_train[indexes1]

X_test  = np.append(X_5_test,X_0_test,0)
indexes2 = np.arange(0,len(X_test))
np.random.shuffle(indexes2)
X_test = X_test[[indexes2]]

y_test = np.append(y_5_test,y_0_test,0)
y_test = y_test[indexes2]

#~~~~~~~~~~~~ Normalizando os Dados ~~~~~~~~~~~~

X_train = X_train/255
X_test = X_test/255


#-------------------------------------- CROAÇÃO E OTIMIZAÇÃO DA REDE NEURAL CONVOLUCIONAL  --------------------------------------

#~~~~~~~~~~~~ Criando Tabela ~~~~~~~~~~~~

""" O intuito desta tabela é listar as diferentes combinações de hiperparâmetros para construção de uma Rede Neural Convolucional (CNN) 
    Essa parte do código demora bastante para rodar pois estão sendo geradas 360 diferentes CNN's cada uma com parametros diferentes."""

batchsizes = 2**np.arange(5,10)
acts = ['relu','sigmoid','softmax','softplus','softsign','tanh', 'selu','elu','exponential']
optmizers = ['SGD','RMSprop','Adam','Adadelta','Adagrad','Adamax','Nadam','Ftrl']
combinacoes = [[a,b,c,0,0] for a in batchsizes for b in acts for c in optmizers]
tabela = pd.DataFrame(combinacoes, columns = ['Batch','Activation','Optmizers','Loss','Accuracy'])

#~~~~~~~~~~~~ Treinando as CNN ~~~~~~~~~~~~

lista = np.arange(0,len(tabela))
np.random.shuffle(lista)

for i in lista:
  # Criando a Rede Neural Convulucional
  batch = tabela.loc[i,'Batch']
  act = tabela.loc[i,'Activation']
  opt = tabela.loc[i,'Optmizers']
  input_shape = (batch, 28, 28)
  model = models.Sequential()
  model.add(layers.Conv1D(batch, 3, activation=act, input_shape=(28,28)))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(64, 3, activation=act))
  model.add(layers.MaxPooling1D(2))
  model.add(layers.Conv1D(64, 3, activation=act))
  model.add(layers.Flatten())
  model.add(layers.Dense(64, activation=act))
  model.add(layers.Dense(10))
  model.compile(optimizer=opt,
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

  history = model.fit(X_train, y_train, epochs=10, verbose=None)
  tabela.loc[i,'Loss'] = history.history['loss'][-1]
  tabela.loc[i,'Accuracy'] = history.history['accuracy'][-1]
 
#~~~~~~~~~~~~ Salvando a Tabela com os Resultados ~~~~~~~~~~~~
tabela.to_excel('Tabela hiperparametros MNIST.xlsx')

""" Analisando a Tabela podemos ver que os parametros que levaram á melhor acurácia com maior frequência foram:
    batchsize = 256, Activation = 'tanh', Optmizer = 'Nadam'. """

# Obtendo o Resultado para o Banco de Teste:

batch = 256
act = 'tanh'
opt = 'Nadam'
input_shape = (batch, 28, 28)
model = models.Sequential()
model.add(layers.Conv1D(batch, 3, activation=act, input_shape=(28,28)))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation=act))
model.add(layers.MaxPooling1D(2))
model.add(layers.Conv1D(64, 3, activation=act))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation=act))
model.add(layers.Dense(2))
model.compile(optimizer=opt,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, verbose=None)

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
