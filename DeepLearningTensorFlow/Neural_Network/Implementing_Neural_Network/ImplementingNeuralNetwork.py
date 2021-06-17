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
#Inputs to a neural network are usually not considered the actual transformative layers. 
#They are merely placeholders for data. In Keras, an input for a neural network can be specified with a tf.keras.layers.InputLayer object.
#The following code initializes an input layer for a DataFrame my_data that has 15 columns
## from tensorflow.keras.layers import InputLayer
## my_input = InputLayer(input_shape=(15,))
#Notice that the input_shape parameter has to have its first dimension equal to the number of features in the data. 
#You don’t need to specify the second dimension: the number of samples or batch size.
#The following code avoids hard-coding with using the .shape property of the my_data DataFrame
## num_features = my_data.shape[1] 
#without hard-coding
## my_input = tf.keras.layers.InputLayer(input_shape=(num_features,)) 
#The following code adds this input layer to a model instance my_model:
## my_model.add(my_input)
#The following code prints a useful summary of a model instance my_model
## print(my_model.summary())
############################################################################################################################################################

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers


def design_model(features):
  model = Sequential(name = "my_first_model")
  #In the design_model() function, create a variable called num_features and assign it the number of columns 
  #in the features DataFrame using the .shape property.
  num_features = features.shape[1]
  #In the design_model() function:
#create a variable called input
#assign input an instance of InputLayer
#set the first dimension of the input_shape parameter #equal to num_features
#Then add the input layer to the model.
  input = layers.InputLayer(input_shape=(num_features,))
  model.add(input)
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
#print the summary of model
print(model.summary())



##########################################################################################################################################################
#Neural Model: Output Layer
#The output layer shape depends on your task. In the case of regression, we need one output for each sample. For example, 
#if your data has 100 samples, you would expect your output to be a vector with 100 entries - a numerical prediction for each sample.
#In our case, we are doing regression and wish to predict one number for each data point: 
#the medical cost billed by health insurance indicated in the charges column in our data. Hence, our output layer has only one neuron.
#The following command adds a layer with one neuron to a model instance my_model

##from tensorflow.keras.layers import Dense
##my_model.add(Dense(1))

#Notice that you don’t need to specify the input shape of this layer since Tensorflow with Keras can automatically infer its shape from the previous layer.
############################################################################################################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense


def design_model(features):
  model = Sequential(name = "my_first_model")
  num_features = features.shape[1]
  input = InputLayer(input_shape=(num_features,))
  model.add(input) #add the input layer
  #In a single command, create and add an output layer to the model instance model as an instance of tensorflow.keras.layers.Dense
  model.add(Dense(1))
  return model


dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42)

 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)
print(model.summary())



###################################################################################################################################################
#Neural Network: Hidden Layers
#So far we have added one input layer and one output layer to our model. 
#If you think about it, our model currently represents a linear regression. 
#To capture more complex or non-linear interactions among the inputs and outputs neural networks, we’ll need to incorporate hidden layers.
#The following command adds a hidden layer to a model instance my_model

##from tensorflow.keras.layers import Dense
##my_model.add(Dense(64, activation='relu'))

#We chose 64 (2^6) to be the number of neurons since it makes optimization more efficient due to the binary nature of computation.
#With the activation parameter, we specify which activation function we want to have in the output of our hidden layer. 
#There are a number of activation functions such as softmax, sigmoid, but ReLU (relu) (Rectified Linear Unit) is very effective 
#in many applications and we’ll use it here.
#Adding more layers to a neural network naturally increases the number of parameters to be tuned. 
#With every layer, there are associated weight and bias vectors.

###########################################################################################################################################################


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense


def design_model(features):
  model = Sequential(name = "my_first_model")
  input = InputLayer(input_shape=(features.shape[1],))
  #add the input layer
  model.add(input) 
  #In the design_model() function, in a single     #command, add a new hidden layer to the model instance #model with the following parameters:
#128 hidden units
#a relu activation function
  model.add(Dense(128, activation='relu'))

  #adding an output layer to our model
  model.add(Dense(1)) 
  return model


dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 

 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)

#print the model summary here
print(model.summary())




################################################################################################################################################
#Optimizers
#As we mentioned, our goal is for the network to effectively adjust its weights or parameters in order to reach the best performance. 
#Keras offers a variety of optimizers such as SGD (Stochastic Gradient Descent optimizer), Adam, RMSprop, and others.
#We’ll start by introducing the Adam optimizer

##from tensorflow.keras.optimizers import Adam
##opt = Adam(learning_rate=0.01)

#The learning rate determines how big of jumps the optimizer makes in the parameter space (weights and bias) and 
#it is considered a hyperparameter that can be also tuned. While model parameters are the ones that the model uses to make predictions, 
#hyperparameters determine the learning process (learning rate, number of iterations, optimizer type).
#If the learning rate is set too high, the optimizer will make large jumps and possibly miss the solution. 
#On the other hand, if set too low, the learning process is too slow and might not converge to a desirable solution with the allotted time. 
#Here we’ll use a value of 0.01, which is often used.
#Once the optimizer algorithm is chosen, a model instance my_model is compiled with the following code

##my_model.compile(loss='mse',  metrics=['mae'], optimizer=opt)

#loss denotes the measure of learning success and the lower the loss the better the performance. 
#In the case of regression, the most often used loss function is the Mean Squared Error mse 
#(the average squared difference between the estimated values and the actual value).
#Additionally, we want to observe the progress of the Mean Absolute Error (mae) while training the model because MAE can give us a better 
#idea than mse on how far off we are from the true values in the units we are predicting. In our case, 
#we are predicting charge in dollars and MAE will tell us how many dollars we’re off, on average, from the actual values as the network is being trained.
###########################################################################################################################################################


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


def design_model(features):
  model = Sequential(name = "my_first_model")
  input = InputLayer(input_shape=(features.shape[1],))
   #add an input layer
  model.add(input)
  #add a hidden layer with 128 neurons
  model.add(Dense(128, activation='relu')) 
  #add an output layer
  model.add(Dense(1)) 
  #In the design_model() function, create an instance of Adam optimizer with 0.01 learning rate and assign the result to a variable called opt.

  opt = Adam(learning_rate=0.01)

  #In the design_model() function, use the .compile()  method to compile the model instance model with:
#the mse loss, mae metrics, opt as the optimizer
  model.compile(loss='mse', metrics=['mae'], optimizer=opt)
  return model


dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)
print(model.summary())


###################################################################################################################################################
#Training and Evaluating the Model
#Now that we built the model we are ready to train the model using the training data.
#The following command trains a model instance my_model using training data my_data and training labels my_labels

##my_model.fit(my_data, my_labels, epochs=50, batch_size=3, verbose=1)

#model.fit() takes the following parameters:
#my_data is the training data set.
#my_labels are true labels for the training data points
#epochs refers to the number of cycles through the full training dataset. 
#Since training of neural networks is an iterative process, you need multiple passes through data. Here we chose 50 epochs
#batch_size is the number of data points to work through before updating the model parameters. It is also a hyperparameter that can be tuned.
#verbose = 1 will show you the progress bar of the training.
#When the training is finalized, we use the trained model to predict values for samples that the training procedure haven’t seen: the test set.
#The following commands evaluates the model instance my_model using the test data my_data and test labels my_labels

##val_mse, val_mae = my_model.evaluate(my_data, my_labels, verbose = 0)

#In our case, model.evaluate() returns the value for our chosen loss metrics (mse) and for an additional metrics (mae).
#So what is the final result? We should get ~$3884.21. This means that on average we’re off with our prediction by around 3800 dollars. 
#Is that a good result or a bad result?
######################################################################################################################################################

import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

tensorflow.random.set_seed(35) #for the reproducibility of results

def design_model(features):
  model = Sequential(name = "my_first_model")
  #without hard-coding
  input = InputLayer(input_shape=(features.shape[1],)) 
  #add the input layer
  model.add(input) 
  #add a hidden layer with 128 neurons
  model.add(Dense(128, activation='relu')) 
  #add an output layer to our model
  model.add(Dense(1)) 
  opt = Adam(learning_rate=0.1)
  model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
  return model

dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)
print(model.summary())

#fit the model using 40 epochs and batch size 1
model.fit(features_train, labels_train, epochs = 40, batch_size = 1, verbose = 1)
 
#evaluate the model on the test data
val_mse, val_mae = model.evaluate(features_test, labels_test, verbose = 0)
 
print("MAE: ", val_mae)


##########################################################################################################################################################
#Summary of full code
#Here is it
##########################################################################################################################################################

import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

tensorflow.random.set_seed(35) #for the reproducibility of results

def design_model(features):
  model = Sequential(name = "my_first_model")
  #without hard-coding
  input = InputLayer(input_shape=(features.shape[1],)) 
  #add the input layer
  model.add(input) 
  #add a hidden layer with 128 neurons
  model.add(Dense(128, activation='relu')) 
  #add an output layer to our model
  model.add(Dense(1)) 
  opt = Adam(learning_rate=0.1)
  model.compile(loss='mse',  metrics=['mae'], optimizer=opt)
  return model

dataset = pd.read_csv('insurance.csv') #load the dataset
features = dataset.iloc[:,0:6] #choose first 7 columns as features
labels = dataset.iloc[:,-1] #choose the final column for prediction

features = pd.get_dummies(features) #one-hot encoding for categorical variables
#split the data into training and test data
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.33, random_state=42) 
 
#standardize
ct = ColumnTransformer([('standardize', StandardScaler(), ['age', 'bmi', 'children'])], remainder='passthrough')
features_train = ct.fit_transform(features_train)
features_test = ct.transform(features_test)

#invoke the function for our model design
model = design_model(features_train)
print(model.summary())

#fit the model using 40 epochs and batch size 1
model.fit(features_train, labels_train, epochs = 40, batch_size = 1, verbose = 1)
 
#evaluate the model on the test data
val_mse, val_mae = model.evaluate(features_test, labels_test, verbose = 0)
 
print("MAE: ", val_mae)
