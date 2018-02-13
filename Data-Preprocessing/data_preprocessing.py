#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Data Preprocessing

# importer les librairies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importer le dataset
dataset = pd.read_csv('Data.csv')

# creation de la matrix associer au dataset
# : toute les colone | :-1 toute les colone sauf la derniere
X = dataset.iloc[:, :-1].values

# selectionner les ligne de la matrix
Y = dataset.iloc[:, -1]

# remplacer le données manquente par la moyenne de la colonne care les dataset sont normalement distribuer
# Gestion des données manquante
# utilisatioin de la class sklearn.preproccessing import Iùputer
from sklearn import preprocessing
from sklearn.preprocessing import Imputer
# creation de l'objet de la classe
# remplacer dans la matrixe les donc les valeur est NAN
imputer = Imputer (missing_values='NaN', strategy='mean', axis=0)
# lié les donnée imputer a la varibale independante (X)
# remplacer les donnée manquante dans les matrixe d'indice  1 et 2 selectioner les indice 1 et 2 (soit [1.2] = 1:3)
# remplacer les variable de valeur nan par la moyenne ds colones d'indice 1 et 2 
imputer.fit(X[:, 1:3])
# remplacer les variables manquantes avec les methide transform 
X[:, 1:3] =  imputer.transform(X[:, 1:3])

# les variables categorique : ce sont des variables qui ne sont pas numerique ,
# gestion des variable categoric pour pourvoir les utiliser dans les finctions
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# creation des objet de chaque classe
labelencoder_X = LabelEncoder()
# liaison a la colonne a la quelle noous allons transformer (remplacer les pays par les veleurs num 0.1.2)
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
# encoder une colonne sous forme de demi variable
onehotencoder = OneHotEncoder(categorical_features = [0])
# creation d'une table avec les les 3 premier variable (france, espagne ,allemagne)
X = onehotencoder.fit_transform(X).toarray()
# encoder la variable y en 0 et 1 no = 0 et yes = 1
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

#training seat et test set
# construire le model sur le traning set et tester le performance sur le test set
from sklearn.model_selection import train_test_split
# X_train : matrixe de variable independante du traning set
# X_test : matrixe de variable independante du test set
# Y_train : vecteur de variable independante du traning set
# Y_test : vecteur de variable independante du test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Future scaling : mettre tous nos variables sur la meme echelle (Mise à l'échelle future)
# nous le feson pour qu'une variable n'ecrase pas l'autre dans l'equetion de machine learning
from sklearn.preprocessing import StandardScaler
# creation de l'objet StandardScaler
sc = StandardScaler()
# pour le calcule nous av*ons besoin de la moyenne et de l'ecart type
# fiter (transformer le X_train)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


