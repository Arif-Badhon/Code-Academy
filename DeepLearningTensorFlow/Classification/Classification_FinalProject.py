#dataset:https://www.kaggle.com/andrewmvd/heart-failure-clinical-data

import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, Normalizer
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report
from tensorflow.keras.utils import to_categorical
import numpy as np

########...Loading_Data...##########
#Read the data
data = pd.read_csv('heart_failure.csv')

#see the info of the data
print(data.info())

#print the distribution of death event column in data
#this column is needed to be predict
print('Classes and number of values in the dataset',Counter(data['death_event']))

#extract the label column death event
y = data['death_event']

#extract the features column
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

########....Data_Processing......###########

#convert categorical features to one-hot encoding
x = pd.get_dummies(x)

#split the data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

#initialize a column transfer
ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

#Use the ColumnTransformer.fit_transform() function to train the scaler instance ct on the training data X_train and assign the result back to X_train
X_train = ct.fit_transform(X_train)

#Use the ColumnTransformer.transform() to scale the test data instance X_test using the trained scaler ct, and assign the result back to X_test.
X_test = ct.transform(X_test)

###.....Prepare labels for classification...#####
#Initialize an instance of LabelEncoder
le = LabelEncoder()

#Fit the encoder instance to training label
Y_train = le.fit_transform(Y_train.astype(str))

#Encode the test labels
Y_test = le.transform(Y_test.astype(str))

#Transform Y_ into a binary vector
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

####......Design the model....########
#Initialize the model
model = Sequential()

#Create an input layer instance for model
model.add(InputLayer(input_shape=(X_train.shape[1],)))
#create a hidden layer
model.add(Dense(12, activation = 'relu'))
#create output layer
model.add(Dense(2, activation='softmax'))

#Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

####.....Train and Evaluate the Model.....######

model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose = 1)

#model evaluate
loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print("Loss", loss, "Accuracy", acc)

####.....Generating a classification report...#####
#predictaion 
y_estimate = model.predict(X_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)
y_true = np.argmax(Y_test, axis=1)

print(classification_report(y_true, y_estimate))
