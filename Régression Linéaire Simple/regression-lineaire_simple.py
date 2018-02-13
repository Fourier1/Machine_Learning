#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
# X = Experience
# Y = Salaires
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1]

# PAS DE DOONNEES MANQUANTES
# Gestion des données manquante
# from sklearn.preprocessing import Imputer
# imputer = Imputer (missing_values='NaN', strategy='mean', axis=0)
# imputer.fit(X[:, 1:3])
# X[:, 1:3] =  imputer.transform(X[:, 1:3])

# PAS DE VARIABLE CATEGORIQUES (DES STRING)
# gestion des variable categoric pour pourvoir les utiliser dans les finctions
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# labelencoder_X = LabelEncoder()
# X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# onehotencoder = OneHotEncoder(categorical_features = [0])
# X = onehotencoder.fit_transform(X).toarray()
# labelencoder_Y = LabelEncoder()
# Y = labelencoder_Y.fit_transform(Y)

#training seat et test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size =1.0/3, random_state = 0)

# PAS BESOIN DE METTRE LES VARIABLES SUR LA MEME ECHELLE
# Future scaling : mettre tous nos variables sur la meme echelle (Mise à l'échelle future)
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X_train = sc.fit_transform(X_tain)
# X_test = sc.transform(X_test)

# construire le model de regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# faire des nouvelle predictions 
# entrer les valeurs des variables independante pour les que on veus predire les variables dependante
# y_pred = x_test
Y_pred = regressor.predict(X_test)
# predire le salaire d'un employer aiyant 15 année d'experiances (qui n'a pas de train)
regressor.predict(15)

# Visualiser les resulats de regression lineaire
# utilser la fonction matplotlib
plt.scatter(X_test, Y_test, color='red')
# tracer de la courbe
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
# un titre
plt.title('SALAIRE Vs EXPERIENCE')
# donner un nom a l'axe des X
plt.xlabel('Expérience')
# donner un nom a l'axe des Y
plt.ylabel('Salaire')

# afficher le graphe
plt.show()