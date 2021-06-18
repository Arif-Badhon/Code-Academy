###########################################.......................CROSS_ENTROPY.....................#####################################################

#cross-entropy, an important concept for evaluating classification model training. 
#Cross-entropy is a score that summarizes the average difference between the actual and predicted probability distributions for all classes. 
#The goal is to minimize the score, with a perfect cross-entropy value is 0.

from sklearn.metrics import log_loss

#consider a problem with three classes, each having three examples in the data classified in class 1, class 2, and class 3, respectively. 
#They are represented with one-hot encoding.
#Let the true distribution for each example be:

#the first class is set to probability 1, all others are 0; this example belongs to class #1
ex_1_true = [1, 0, 0] 
#the second class is set to probability 1, all others are 0;this example belongs to class #2
ex_2_true = [0, 1, 0] 
#the third class is set to probability 1, all others are 0;this example belongs to class #3
ex_3_true = [0, 0, 1] 

#Now imagine a predictive model that gave us the following predictions:
#the highest probability is given to class #1
ex_1_predicted = [0.7, 0.2, 0.1] 
#the highest probability is given to class #2
ex_2_predicted = [0.1, 0.8, 0.1] 
#the highest probability is given to class #3
ex_3_predicted = [0.2, 0.2, 0.6] 

#If we compare the true and predicted distributions above, they seem to be rather different numbers, 
#but there is a good pattern here: each example’s predicted distribution gives the highest probability to the label the example actually belongs to. 
#This means the distributions are similar and the cross-entropy should be small. When we calculate cross-entropy for the example above, we get 0.364, 
#which is rather good and close to 0.

#Now, consider a bad predictive model that gives the highest probability to a wrong label every time:
#the highest probability given to class #3, true labels is class #1
ex_1_predicted_bad = [0.1, 0.1, 0.7]
#the highest probability given to class #1, true labels is class #2
ex_2_predicted_bad = [0.8, 0.1, 0.1] 
#the highest probability given to class #1, true labels is class #3
ex_3_predicted_bad = [0.6, 0.2, 0.2] 

#When we calculate the cross-entropy for these examples, we get 2.036, which is rather bad.
#If we take cross-entropy between two identical true distributions, we get perfect probabilities and cross-entropy equal to 0.

true_labels = [ex_1_true, ex_2_true, ex_3_true]
predicted_labels = [ex_1_predicted, ex_2_predicted, ex_3_predicted]
predicted_labels_bad = [ex_1_predicted_bad, ex_2_predicted_bad, ex_3_predicted_bad]

ll = log_loss(true_labels, predicted_labels)
print('Average Log Loss (good prediction): %.3f' % ll)

ll = log_loss(true_labels, predicted_labels_bad)
print('Average Log Loss (bad prediction): %.3f' % ll)

ll = log_loss(true_labels, true_labels)
print('(TODO)Average Log Loss (true prediction): %.3f' % ll)


##########################################################################################################################################################
#Loading and analyzing the data
#Assume we have a dataset, stored in the train_glass.csv (training data) and test_glass.csv (test data) files, about various products made of glass.
#Using the train_glass.csv file, we want to learn a model that can predict which glass item can be constructed given the proportion of 
#various elements such as Aluminium (Al), Magnesium (Mg), and Iron (Fe). We then want to evaluate the model on the test data.
#To load the training data into a pandas DataFrame, we do the following:

##import pandas as pd
##data_train = pd.read_csv("train_glass.csv")

#The following command lists all features with accompanying types about the columns:

##print(data_train.info())

#The output looks something like this:

##   Column    Non-Null Count   Dtype  
##---  ------   --------------   -----  
## 0   Al       300 non-null     float64
## 1   Mg       300 non-null     float64 
## 3   Fe       300 non-null     float64
## 4   item     300 non-null     object

#We see that Al, Mg, and Fe are numeric columns, and item is an object column containing strings. We would like to predict the item column.
#The following commands show us which categories we have in the item column and what their distribution is:

##from collections import Counter
##print('Classes and number of values in the dataset`,Counter(data_train[“item”]))

#which gives something like the following output:
#{‘lamps’: 75, ‘tableware’: 125, 'containers': 100}

#This tells us that we have three categories to predict: “lamps”, “tableware”, and “containers”, and how many samples we have in our training data for each.

#Next, we we need to split our data into features and labels by doing the following:

##train_x = data_train["item"]
##train_y = data_train[[‘Al', ‘Mg’, 'Fe’]]
##############################################################################################################################################################

import pandas as pd
from collections import Counter

#Using pandas, load the air_quality_train.csv into a DataFrame instance called train_data, and load the air_quality_test.csv into a 
#DataFrame instance called test_data.
train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")
#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data['Air_Quality']

#############################################################################################################################################################
#Preparing the data
#When using categorical cross-entropy — the loss function necessary for multiclass classification problems — in TensorFlow with Keras, 
#one needs to convert all the categorical features and labels into one-hot encoding vectors. Previously, when we had features encoded as strings, 
#we used the pandas.get_dummies() function. This works well for features, but it’s not very usable for labels. 
#The problem is that get_dummies() creates a separate column for each category, and you cannot predict for multiple columns.
#A better approach is to convert the label vectors to integers ranging from 0 to the number of classes by using sklearn.preprocessing.LabelEncoder

##from sklearn.preprocessing import LabelEncoder
##le=LabelEncoder()
##train_y=le.fit_transform(train_y.astype(str))
##test_y=le.transform(test_y.astype(str))

#We first fit the transformer to the training data using the LabelEncoder.fit_transform() method, 
#and then fit the trained transformer to the test data using the LabelEncoder.transform() method.
#We can print the resulting mappings with:

##integer_mapping = {l: i for i, l in enumerate(le.classes_)}
##print(integer_mapping)

#we get the following output

##{‘lamps’: 0, ‘tableware': 1, 'containers': 2}. 

#Each category is mapped to an integer, from 0 to 2 (because we have three categories).
#Now that we have labels as integers, we can use a Keras function called to_categorical() to convert them into one-hot-encodings — 
#the format we need for our cross-entropy loss

##train_y = tensorflow.keras.utils.to_categorical(train_y, dtype = ‘int64’)
##test_y = tensorflow.keras.utils.to_categorical(test_y, dtype = ‘int64’)
############################################################################################################################################################


import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
#your code here

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#Use the LabelEncoder.fit_transform() method to encode the label vector y_train into integers and assign the result back to the y_train variable.
y_train = le.fit_transform(y_train.astype(str))
#Use the le.transform() method to encode the label vector y_test into integers, where le is the instance of LabelEncoder trained in the previous step, 
#and assign the result back to y_test.
y_test = le.transform(y_test.astype(str))

#onvert the integer encoded label vector y_train into a one-hot encoding vector and assign the result
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')


###########################################################################################################################################################
#Designing a deep learning model for classification
#To initialize a Keras Sequential model in TensorFlow, we do the following

##from tensorflow.keras.models import Sequential
##my_model = Sequential()

#The process is the following:
#set the input layer
#set the hidden layers
#set the output layer.
#To add the input layer, we use keras.layers.InputLayer the following way:

##from tensorflow.keras.layers import  InputLayer
##my_model.add(InputLayer(input_shape=(data_train.shape[1],)))

#For now, we will only add one hidden layer using keras.layers.Dense

##from tensorflow.keras.layers import  Dense
##my_model.add(Dense(8, activation='relu'))

#This layer has eight hidden units and uses a rectified linear unit (relu) as the activation function.
#Finally, we need to set the output layer. For regression, we don’t use any activation function in the final layer because we needed 
#to predict a number without any transformations. However, for classification, the desired output is a vector of categorical probabilities.
#To have this vector as an output, we need to use the softmax activation function that outputs a vector with elements having values 
#between 0 and 1 and that sum to 1 (just as all the probabilities of all outcomes for a random variable must sum up to 1). 
#In the case of a binary classification problem, a sigmoid activation function can also be used in the output layer but paired with the binary_crossentropy loss.
#Since we have 3 classes to predict in our glass production data, the final softmax layer must have 3 units:

##my_model.add(Dense(3, activation='softmax')) #the output layer is a softmax with 3 units
###########################################################################################################################################################

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()

#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#add a hidden layer
model.add(Dense(10, activation='relu'))
#add an output layer
model.add(Dense(6, activation='softmax'))

########################################################################################################################################################
#Setting the optimizer
#Now that we’ve had a brief introduction to cross-entropy, we’ll see how to use it with our model.
#First, to specify the use of cross-entropy when optimizing the model, we need to set the loss parameter to categorical_crossentropy of the 
#Model.compile() method.
#Second, we also need to decide which metrics to use to evaluate our model. For classification, we usually use accuracy. 
#Accuracy calculates how often predictions equal labels and is expressed in percentages. We will use this metric for our problem.
#Finally, we will use Adam as our optimizer because it’s effective here and is commonly used.
#To compile the model with all the specifications mentioned above we do the following

##my_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#########################################################################################################################################################

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
#your code here

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()
#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#add a hidden layer
model.add(Dense(10, activation='relu'))
#add an output layer
model.add(Dense(6, activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

######################################################################################################################################################
#Train and evaluate the classification model
#To train a model instance my_model on the training data my_data and training labels my_labels we do the following

##my_model.fit(my_data, my_labels, epochs=10, batch_size=1, verbose=1)

#With the command above, we set the number of epochs to 10 and the batch size to 1. To see the progress of the training we set verbose to true (1).
#After the model is trained, we can evaluate it using the unseen test data my_test and test labels test_labels

##loss, acc = my_model.evaluate(my_test, test_labels, verbose=0)

#We take two outputs out of the .evaluate() function:
#the value of the loss (categorical_crossentropy)
#accuracy (as set in the metrics parameter of .compile()).
#########################################################################################################################################################

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
#your code here

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()
#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#add a hidden layer
model.add(Dense(10, activation='relu'))
#add an output layer
model.add(Dense(6, activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train and evaluate the model
model.fit(x_train, y_train, epochs = 20, batch_size = 4, verbose = 1)


#########################################################################################################################################################
#Additional evaluation statistics
#Sometimes having only accuracy reported is not enough or adequate. Accuracy is often used when data is balanced, 
#meaning it contains an equal or almost equal number of samples from all the classes. However, oftentimes data comes imbalanced. 
#For example in medicine, the rate of a disease is low. In these cases, reporting another metric such as F1-score is more adequate.
#Frequently, especially in medicine, false negatives and false positives have different consequences. For example, 
#in medicine, if we generate a false negative it means that we claim a patient doesn’t have a disease, while they actually have it — yikes! 
#Luckily, an F1-score is a helpful way to evaluate our model based on how badly it makes false negative mistakes.
#To observe the F1-score of a trained model instance my_model, amongst other metrics, we use sklearn.metrics.classification_report

##import numpy as np
##from sklearn.metrics import classification_report
##yhat_classes = np.argmax(my_model.predict(my_test), axis = -1)
##y_true = np.argmax(my_test_labels, axis=1)
##print(classification_report(y_true, yhat_classes))

#In the code above we do the following:
#predict classes for all test cases my_test using the .predict() method and assign the result to the yhat_classes variable.
#using .argmax() convert the one-hot-encoded labels my_test_labels into the index of the class the sample belongs to. 
#The index corresponds to our class encoded as an integer.
#use the .classification_report() method to calculate all the metrics.
###########################################################################################################################################################

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
from sklearn.metrics import classification_report
import numpy as np
#your code here

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()
#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#add a hidden layer
model.add(Dense(10, activation='relu'))
#add an output layer
model.add(Dense(6, activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train and evaluate the model
model.fit(x_train, y_train, epochs = 20, batch_size = 16, verbose = 1)

#Using the Model.predict() method, get the predictions for your test data x_test using the trained model instance model.
y_estimate = model.predict(x_test)

#convert the one-hot encoded labels y_estimate into the index of the class each sample in the test data
y_estimate = np.argmax(y_estimate, axis=1)

#convert the one-hot encoded labels y_test into the index of the class each sample in the test data
y_true = np.argmax(y_test, axis=1)
#get additional statistics
print(classification_report(y_true, y_estimate))


######################################################################################################################################################
#Classification loss alternative: sparse crossentropy
#As we saw before, categorical cross-entropy requires that we first integer-encode our categorical labels and then convert them to one-hot encodings 
#using to_categorical(). There is another type of loss – sparse categorical cross-entropy – which is a computationally modified categorical 
#cross-entropy loss that allows you to leave the integer labels as they are and skip the entire procedure of encoding.
#Sparse categorical cross-entropy is mathematically identical to categorical cross-entropy but introduces some computational 
#shortcuts that save time in memory as well as computation because it uses a single integer for a class, rather than a whole vector. 
#This is especially useful when we have data with many classes to predict.
#We can specify the use of the sparse categorical crossentropy in the .compile() method

##model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Note the following changes: we make sure that our labels are just integer encoded using the LabelEncoder() but not converted into one-hot-encodings 
#using .to_categorical(). Hence, we comment out the code that uses .to_categorical()
###########################################################################################################################################################

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
from sklearn.metrics import classification_report
import numpy as np
#your code here

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
#we comment it here because we need only integer labels for
#sparse cross-entropy
#Using the # symbol for comments, comment out the following lines of code (Line 36 and Line 37):

#y_train = tensorflow.keras.utils.to_categorical#(y_train, dtype = 'int64')
#y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()
#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#add a hidden layer
model.add(Dense(10, activation='relu'))
#add an output layer
model.add(Dense(6, activation='softmax'))

#compile the model
#Modify the existing code on the right by changing the model.compile() method to use sparse_categorical_crossentropy as loss
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train and evaluate the model
model.fit(x_train, y_train, epochs = 20, batch_size = 16, verbose = 0)

#get additional statistics
y_estimate = model.predict(x_test, verbose=0)
y_estimate = np.argmax(y_estimate, axis=1)
print(classification_report(y_test, y_estimate))

#####################################################################################################################################################
#Tweak the model
#Now that we have run our code several times, we might be wondering if the model can be further improved.
#The first thing we can try is to increase the number of epochs. Having 20 epochs, as we previously had, is usually not enough. 
#Try changing the number of epochs, for example, to 40 and see what happens. Increasing the number of epochs naturally makes the learning longer, 
#but as you probably observed, the results are often much better.
#Other hyperparameters you might consider changing are: the batch size number of hidden layers number of units per hidden layer the 
#learning rate of the optimizer the optimizer and so on.
######################################################################################################################################################

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
from sklearn.metrics import classification_report
import numpy as np

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')

#design the model
model = Sequential()
#add the input layer
model.add(InputLayer(input_shape=(x_train.shape[1],)))
#add a hidden layer
model.add(Dense(10, activation='relu'))
#add an output layer
model.add(Dense(6, activation='softmax'))

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#train and evaluate the model
#Change the number of epochs from 20 to 30. Rerun the code and observe the results.
model.fit(x_train, y_train, epochs = 30, batch_size = 16, verbose = 0)

#get additional statistics
y_estimate = model.predict(x_test)
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(y_test, axis = 1)
print(classification_report(y_true, y_estimate))

######################################################################################################################################################
#Summary
######################################################################################################################################################

import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
from sklearn.metrics import classification_report
import numpy as np

train_data = pd.read_csv("air_quality_train.csv")
test_data = pd.read_csv("air_quality_test.csv")

#print columns and their respective types
print(train_data.info())
#print the class distribution
print(Counter(train_data["Air_Quality"]))
#extract the features from the training data
x_train = train_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the training data
y_train = train_data["Air_Quality"]
#extract the features from the test data
x_test = test_data[['PM2.5', 'PM10', 'NO', 'NO2', 'NOx', 'NH3', 'CO', 'SO2', 'O3', 'Benzene', 'Toluene', 'Xylene', 'AQI']]
#extract the label column from the test data
y_test = test_data["Air_Quality"]

#encode the labels into integers
le = LabelEncoder()
#convert the integer encoded labels into binary vectors
y_train=le.fit_transform(y_train.astype(str))
y_test=le.transform(y_test.astype(str))
#convert the integer encoded labels into binary vectors
y_train = tensorflow.keras.utils.to_categorical(y_train, dtype = 'int64')
y_test = tensorflow.keras.utils.to_categorical(y_test, dtype = 'int64')










