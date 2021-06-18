import app
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow	import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

#Read Data from csv
dataset = pd.read_csv('admissions_data.csv')

#drop serial no from dataframe
dataset = dataset.drop(['Serial No.'], axis = 1)
print(dataset.head())
# Split up the data into features and labels
#labels are in the last column of the dataset
labels = dataset.iloc[:, -1]
#features span from first column to the last column but not including it
features = dataset.iloc[:, 0:-1]

#Split the dataset into training and test data set
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size = 0.30, random_state = 25)

#The next step is to standardize/normalize your numerical features.
numerical_features = features.select_dtypes(include=['float64', 'int64'])
numerical_columns = numerical_features.columns
 
ct = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

#Fit your instance ct of ColumnTransformer to the training data and at the same time transform it by using the ColumnTransformer.fit_transform() method. 
#Assign the result to a variable called features_train_scaled.
features_train_scaled = ct.fit_transform(features_train)

#ransform your test data instance features_test using the trained ColumnTransformer instance ct. Assign the result to a variable called features_test_scaled.
features_test_scaled = ct.fit_transform(features_test)

#Create an instance of the tensorflow.keras.models.Sequential model called my_model.
my_model = Sequential()

#Create the input layer to the network model using tf.keras.layers.InputLayer with the shape corresponding to the number of features in your dataset. 
#Assign the resulting input layer to a variable called input.
input = InputLayer(input_shape = (features.shape[1], ))

#Add the input layer from the previous step to the model instance my_model.
my_model.add(input)

#Add one keras.layers.Dense hidden layer with any number of hidden units you wish. Use the relu activation function.
my_model.add(Dense(64, activation = "relu"))

#Add a keras.layers.Dense output layer with one neuron since you need a single output for a regression prediction.
my_model.add(Dense(1))

#Print the summary of the model using the Sequential.summary() method.
print(my_model.summary())

#Create an instance of the Adam optimizer with the learning rate equal to 0.01.
opt = Adam(learning_rate = 0.1)

#Compile model
my_model.compile(loss = 'mse', metrics = ['mae'], optimizer = opt)

#Training the model
my_model.fit(features_train_scaled, labels_train, epochs = 50, batch_size = 5, verbose = 1)

#Using the Sequential.evaluate() method, evaluate your trained model on the preprocessed test data set, and with the test labels. 
#Set verbose to 0. Store the result of the evaluation in two variables: res_mse and res_mae.
res_mse, res_mae = my_model.evaluate(features_test_scaled, labels_test, verbose = 0)

#Print your final loss (RMSE) and final metric (MAE) to check the predictive performance on the test set.
print(res_mse, res_mae)
# if you decide to do the Matplotlib extension, you must save your plot in the directory by uncommenting the line of code below

# fig.savefig('static/images/my_plots.png')
