#######DataPreprocessing:One-hotEncodingAndStandardization##########

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer

#load the dataset
dataset = pd.read_csv('insurance.csv') 
#choose first 7 columns as features
features = dataset.iloc[:,0:6] 
#choose the final column for prediction
labels = dataset.iloc[:,-1] 

#one-hot encoding for categorical variables
features = pd.get_dummies(features) 
#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 
 
#normalize the numeric columns using ColumnTransformer
ct = ColumnTransformer([('normalize', Normalizer(), ['age', 'bmi', 'children'])], remainder='passthrough')
#fit the normalizer to the training data and convert from numpy arrays to pandas frame
features_train_norm = ct.fit_transform(features_train) 
#applied the trained normalizer on the test data and convert from numpy arrays to pandas frame
features_test_norm = ct.transform(features_test) 

#ColumnTransformer returns numpy arrays. Convert the features to dataframes
features_train_norm = pd.DataFrame(features_train_norm, columns = features_train.columns)
features_test_norm = pd.DataFrame(features_test_norm, columns = features_test.columns)


#create new column transformer instance
my_ct = ColumnTransformer([('scale', StandardScaler(),['age', 'bmi', 'children'])], remainder='passthrough')

#Use the .fit_transform() method of my_ct to fit the column transformer to the features_train DataFrame and at 
#the same time transform it. Assign the result to a variable called features_train_scale.
features_train_scale = my_ct.fit_transform(features_train) 

#Use the .transform() method to transform the trained column transformer my_ct to the features_test DataFrame. 
#Assign the result to a variable called features_test_scale.
features_test_scale = my_ct.transform(features_test)

#Transform the features_train_scale NumPy array back to a DataFrame using pd.DataFrame() and assign the result back to a 
#variable called features_train_scale. For the columns attribute use the .columns property of features_train.
features_train_scale = pd.DataFrame(features_train_scale, columns = features_train.columns)

#Transform the features_test_scale NumPy array back to DataFrame using pd.DataFrame() and assign the result back to a 
#variable called features_test_scale. For the columns attribute use the .columns property of features_test.
features_test_scale = pd.DataFrame(features_test_scale, columns = features_test.columns)

#print the summary statistics
print(features_train_scale.describe())
print(features_test_scale.describe())


########################################################################################################################
#Now that we have our data preprocessed we can start building the neural network model. 
#The most frequently used model in TensorFlow is Keras Sequential. A sequential model, as the name suggests, 
#allows us to create models layer-by-layer in a step-by-step fashion. This model can have only one input tensor and only one output tensor.

###############################  Sequential Model   ####################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def design_model(features):
  model = Sequential()
  return model
dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 
#split the data into training and test data
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)

#print the layers
print(model.layers)


##########################################################################################################################################################
#Neural network model: layers
#Layers are the building blocks of neural networks and can contain 1 or more neurons. 
#Each layer is associated with parameters: weights, and bias, that are tuned during the learning. 
#A fully-connected layer in which all neurons connect to all neurons in the next layer is created the following way in TensorFlow
##########################################################################################################################################################

import tensorflow as tf
from tensorflow.keras import layers
layer = layers.Dense(3) #3 is the number we chose

print(layer.weights) #we get empty weight and bias arrays because tensorflow doesn't know what the shape is of the input to this layer

# 1338 samples, 11 features as in our dataset
input = tf.ones((5000, 21)) 
#replace 1338 (the number of samples) with 5000 After you run the code, you will see that the weight and bias matrices didn’t change their shape
#replace 11 (number of features) with 21.
#Changing the number of features of the input data will modify the weight matrix. If you look into the diagram, 
#the dimensions of a weight matrix are features and output. Since our features changed from 11 to 21, 
#the weight matrix will have a new shape (21, 3). The bias array does not change its shape by changing the number of features.


# a fully-connected layer with 3 neurons
layer = layers.Dense(10)
#replace 3 (number of neurons/outputs) with 10 where layer is defined.
#Changing the number of neurons in a layer changes the shape of the weight matrix from (21,3) to (21, 10), and bias from (3,) to (10,).

# calculate the outputs
output = layer(input) 
# print the weights
print(layer.weights) 


########################################################################################################################################################
#Neural Network Model: Input Layer
#
