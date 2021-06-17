#####################################################################################################################################################
#Using a validation set for hyperparameter tuning
#Using the training data to choose hyperparameters might lead to overfitting to the training data meaning the model 
#learns patterns specific to the training data that would not apply to new data. For that reason, 
#hyperparameters are chosen on a held-out set called validation set . 
#In TensorFlow Keras, validation split can be specified as a parameter in the .fit() function

##my_model.fit(data, labels, epochs = 20, batch_size = 1, verbose = 1,  validation_split = 0.2)

#where validation_split is a float between 0 and 1, denoting a fraction of the training data to be used as validation data. 
#In the example above, 20% of the data would be allocated for validation. It is usually a small fraction of the training data. 
#The model will set apart this fraction of the training data, will not train on it, and 
#will evaluate the loss and any model metrics on this data at the end of each epoch.
######################################################################################################################################################

#see neural network folder file for more details
from model import design_model, features_train, labels_train 

model = design_model(features_train, learning_rate = 0.01)
#Use the .fit() function to fit the model instance model to the training data features_train and training 
#features labels_train with 40 epochs, batch size set to 8, verbose set to true (1), and validation split set to 33%.

model.fit(features_train, labels_train, epochs = 40, batch_size = 8, verbose = 1, validation_split = 0.33)


########################################################################################################################################################
#Manual Tuning: Learning Rate
#Neural networks are trained with the gradient descent algorithm and one of the most important hyperparameters 
#in the network training is the learning rate. The learning rate determines how big of a change you apply to the network 
#weights as a consequence of the error gradient calculated on a batch of training data.
#A larger learning rate leads to a faster learning process at a cost to be stuck in a suboptimal solution (local minimum). 
#A smaller learning rate might produce a good suboptimal or global solution, but it will take it much longer to converge. 
#In the extremes, a learning rate too large will lead to an unstable learning process oscillating over the epochs. 
#A learning rate too small may not converge or get stuck in a local minimum.
#It can be helpful to test different learning rates as we change our hyperparameters. A learning rate of 1.0 leads to oscillations, 
#0.01 may give us a good performance, while 1e-07 is too small and almost nothing is learned within the allotted time.
#########################################################################################################################################################

from model import design_model, features_train, labels_train 
import matplotlib.pyplot as plt

def fit_model(f_train, l_train, learning_rate, num_epochs, bs):
    #build the model
    model = design_model(f_train, learning_rate)
    #train the model on the training data
    history = model.fit(f_train, l_train, epochs = num_epochs, batch_size = bs, verbose = 0, validation_split = 0.2)
    # plot learning curves
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('lrate=' + str(learning_rate))
    plt.legend(loc="upper right")


#make a list of learning rates to try out
learning_rates = [1E-3, 1E-4, 1E-7, 0.01]
#fixed number of epochs
num_epochs = 100
#fixed number of batches
batch_size = 10 

for i in range(len(learning_rates)):
  plot_no = 420 + (i+1)
  plt.subplot(plot_no)
  fit_model(features_train, labels_train, learning_rates[i], num_epochs, batch_size)

plt.tight_layout()
plt.show()
plt.savefig('static/images/my_plot.png')
print("See the plot on the right with learning rates", learning_rates)
import app #don't worry about this. This is to show you the plot in the browser.


######################################################################################################################################################
#Manual tuning: batch size
#The batch size is a hyperparameter that determines how many training samples are seen before updating the network’s parameters (weight and bias matrices).
#When the batch contains all the training examples, the process is called batch gradient descent. 
#If the batch has one sample, it is called the stochastic gradient descent. And finally, 
#when 1 < batch size < number of training points, is called mini-batch gradient descent. 
#An advantage of using batches is for GPU computation that can parallelize neural network computations.
#How do we choose the batch size for our model? On one hand, a larger batch size will provide our model with better gradient estimates and 
#a solution close to the optimum, but this comes at a cost of computational efficiency and good generalization performance. 
#On the other hand, smaller batch size is a poor estimate of the gradient, but the learning is performed faster. 
#Finding the “sweet spot” depends on the dataset and the problem, and can be determined through hyperparameter tuning.
#For this experiment, we fix the learning rate to 0.01 and try the following batch sizes: 1, 2, 10, and 16. 
#Notice how small batch sizes have a larger variance (more oscillation in the learning curve).
#Want to improve the performance with a larger batch size? A good trick is to increase the learning rate!
#########################################################################################################################################################


from model import features_train, labels_train, design_model
import matplotlib.pyplot as plt

def fit_model(f_train, l_train, learning_rate, num_epochs, batch_size, ax):
    model = design_model(features_train, learning_rate)
    #train the model on the training data
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size = batch_size, verbose=0, validation_split = 0.3)
    # plot learning curves
    ax.plot(history.history['mae'], label='train')
    ax.plot(history.history['val_mae'], label='validation')
    ax.set_title('batch = ' + str(batch_size), fontdict={'fontsize': 8, 'fontweight': 'medium'})
    ax.set_xlabel('# epochs')
    ax.set_ylabel('mae')
    ax.legend()

#fixed learning rate
##In the previous checkpoint, you might have noticed bad performance for larger batch sizes (32 and 64). 
#When having performance issues with larger batches it might help to increase the learning rate. 
#Modify the value for the learning rate by assigning 0.1 to the learning_rate variable. Rerun the code and observe the plots.
 
learning_rate = 0.1 
#fixed number of epochs
num_epochs = 100
#we choose a number of batch sizes to try out
#Modify the batches list to include the following batch sizes: 4, 32, 64. Rerun the code and observe the plots.
batches = [4, 32, 64] 
print("Learning rate fixed to:", learning_rate)

#plotting code
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex='col', sharey='row',
                        gridspec_kw={'hspace': 0.7, 'wspace': 0.4}) #preparing axes for plotting
axes = [ax1, ax2, ax3]

#iterate through all the batch values
for i in range(len(batches)):
  fit_model(features_train, labels_train, learning_rate, num_epochs, batches[i], axes[i])

plt.savefig('static/images/my_plot.png')
print("See the plot on the right with batch sizes", batches)
import app #don't worry about this. This is to show you the plot in the browser.



##################################################################################################################################################
#Manual tuning: epochs and early stopping
#Being an iterative process, gradient descent iterates many times through the training data. 
#The number of epochs is a hyperparameter representing the number of complete passes through the training dataset. 
#This is typically a large number (100, 1000, or larger). If the data is split into batches, in one epoch the optimizer will see all the batches.
#How do you choose the number of epochs? Too many epochs can lead to overfitting, and too few to underfitting. 
#One trick is to use early stopping: when the training performance reaches the plateau or starts degrading, the learning stops.
#We know we are overfitting because the validation error at first decreases but eventually starts increasing. 
#The final validation MAE is ~3034, while the training MAE is ~1000. That’s a big difference. 
#We see that the training could have been stopped earlier (around epoch 50).
#We can specify early stopping in TensorFlow with Keras by creating an EarlyStopping callback and adding it as a parameter when we fit our model.

##from tensorflow.keras.callbacks import EarlyStopping
##stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=40)
##history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size=16, verbose=0, validation_split=0.2, callbacks=[stop])

#monitor = val_loss, which means we are monitoring the validation loss to decide when to stop the training
#mode = min, which means we seek minimal loss
#patience = 40, which means that if the learning reaches a plateau, it will continue for 40 more epochs in case the plateau leads to improved performance
####################################################################################################################################################


from model import features_train, labels_train, design_model
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

def fit_model(f_train, l_train, learning_rate, num_epochs):
    #build the model: to see the specs go to model.pyl we increased the number of hidden neurons
    #in order to introduce some overfitting
    model = design_model(features_train, learning_rate) 
    #train the model on the training data
    #In the fit_model() method, just before calling model.fit(), create an instance of EarlyStopping that monitors the validation loss (val_loss), 
    #seeks minimal loss, that is verbose, and has patience equal to 20. Assign the result to a variable called es.
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
    #Now that you have an instance of EarlyStopping assigned to es, you need to pass the instance as a callback function to the .fit() method. 
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= 16, verbose=0, validation_split = 0.2, callbacks = [es])
    return history

    
#using the early stopping in fit_model
learning_rate = 0.1
num_epochs = 500
history = fit_model(features_train, labels_train, learning_rate, num_epochs)

#plotting
fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 1, 'wspace': 0.5}) 
(ax1, ax2) = axs
ax1.plot(history.history['loss'], label='train')
ax1.plot(history.history['val_loss'], label='validation')
ax1.set_title('lrate=' + str(learning_rate))
ax1.legend(loc="upper right")
ax1.set_xlabel("# of epochs")
ax1.set_ylabel("loss (mse)")

ax2.plot(history.history['mae'], label='train')
ax2.plot(history.history['val_mae'], label='validation')
ax2.set_title('lrate=' + str(learning_rate))
ax2.legend(loc="upper right")
ax2.set_xlabel("# of epochs")
ax2.set_ylabel("MAE")

print("Final training MAE:", history.history['mae'][-1])
print("Final validation MAE:", history.history['val_mae'][-1])

plt.savefig('static/images/my_plot.png')
import app #don't worry about this. This is to show you the plot in the browser.


############################################################################################################################################################
#Manual tuning: changing the model
#We saw in the previous exercise that if you have a big model and you train too long, you might overfit. 
#Let us see the opposite - having a too simple model.
#In the code on the right, we compare a one-layer neural network and a model with a single hidden layer. The models look like this

##def one_layer_model(X, learning_rate):
   ##...
   ##model.add(input) 
   ##model.add(layers.Dense(1))
   ##...

#and

##def more_complex_model(X, learning_rate):
##    ...
##    model.add(input)
##    model.add(layers.Dense(64, activation='relu'))
##    model.add(layers.Dense(1))

#When we run the learning we get the learning curves depicted on the far right.
#If you observe Plot #1 for the model with no hidden layers you will see an issue: 
#the validation curve is below the training curve. This means that the training curve can get better at some point, 
#but the model complexity doesn’t allow it. This phenomenon is called underfitting. 
#You can also notice that no early stopping occurs here since the performance of this model is bad all the time.
#Plot #2 is for the model with a single hidden layer. You can observe a well-behaving curve with the early stopping 
#at epoch 38 and we have a much better result. Nice!
#How do we choose the number of hidden layers and the number of units per layer? 
#That is a tough question and there is no good answer. The rule of thumb is to start with one hidden layer and add as many units as 
#we have features in the dataset. However, this might not always work. We need to try things out and observe our learning curve.
##############################################################################################################################################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from model import features_train, labels_train

def more_complex_model(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    #Decrease the number of hidden units of the model from 64 to 8 in the more_complex_model() method.
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def one_layer_model(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def fit_model(model, f_train, l_train, learning_rate, num_epochs):
    #train the model on the training data
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience = 20)
    history = model.fit(features_train, labels_train, epochs=num_epochs, batch_size= 2, verbose=0, validation_split = 0.2, callbacks = [es])
    return history

def plot(history):
    # plot learning curves
    fig, axs = plt.subplots(1, 2, gridspec_kw={'hspace': 1, 'wspace': 0.8}) 
    (ax1, ax2) = axs
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_title('lrate=' + str(learning_rate))
    ax1.legend(loc="upper right")
    ax1.set_xlabel("# of epochs")
    ax1.set_ylabel("loss (mse)")

    ax2.plot(history.history['mae'], label='train')
    ax2.plot(history.history['val_mae'], label='validation')
    ax2.set_title('lrate=' + str(learning_rate))
    ax2.legend(loc="upper right")
    ax2.set_xlabel("# of epochs")
    ax2.set_ylabel("MAE")
    print("Final training MAE:", history.history['mae'][-1])
    print("Final validation MAE:", history.history['val_mae'][-1])

learning_rate = 0.1
num_epochs = 200

#fit the more simple model
print("Results of a one layer model:")
history1 = fit_model(one_layer_model(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history1)
plt.savefig('static/images/my_plot1.png')

#fit the more complex model
print("Results of a model with hidden layers:")
history2 = fit_model(more_complex_model(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
plot(history2)
plt.savefig('static/images/my_plot2.png')

import app #don't worry about this. This is to show you the plot in the browser.


#####################################################################################################################################################
#Towards automated tuning: grid and random search
#So far we’ve been manually setting and adjusting hyperparameters to train and evaluate our model. 
#If we didn’t like the result, we changed the hyperparameters to some other values. However, this is rather cumbersome; 
#it would be nice if we could make these changes in a systematic and automated way. Fortunately, 
#there are some strategies for automated hyperparameter tuning, including the following two.
#Grid search, or exhaustive search, tries every combination of desired hyperparameter values. 
#If, for example, we want to try learning rates of 0.01 and 0.001 and batch sizes of 10, 30, and 50, 
#grid search will try six combinations of parameters (0.01 and 10, 0.01 and 30, 0.01 and 50, 0.001 and 10, and so on). 
#This obviously gets very computationally demanding when we increase the number of values per hyperparameter or the number of hyperparameters we want to tune.
#On the other hand, Random Search goes through random combinations of hyperparameters and doesn’t try them all.
#Grid search in Keras
#To use GridSearchCV from scikit-learn for regression we need to first wrap our neural network model into a KerasRegressor:

##model = KerasRegressor(build_fn=design_model)

#Then we need to setup the desired hyperparameters grid (we don’t use many values for the sake of speed)

##batch_size = [10, 40]
##epochs = [10, 50]
##param_grid = dict(batch_size=batch_size, epochs=epochs)

#Finally, we initialize a GridSearchCV object and fit our model to the data

##grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False))
##grid_result = grid.fit(features_train, labels_train, verbose = 0)

#Notice that we initialized the scoring parameter with scikit-learn’s .make_scorer() method. 
#We’re evaluating our hyperparameter combinations with a mean squared error making sure that greater_is_better is set to False since 
#we are searching for a set of hyperparameters that yield us the smallest error.
#Randomized search in Keras
#We first change our hyperparameter grid specification for the randomized search in order to have more options

##param_grid = {'batch_size': sp_randint(2, 16), 'nb_epoch': sp_randint(10, 100)}

#Randomized search will sample values for batch_size and nb_epoch from uniform distributions in the interval [2, 16] and [10, 100], 
#respectively, for a fixed number of iterations. In our case, 12 iterations

##grid = RandomizedSearchCV(estimator = model, param_distributions=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), n_iter = 12)

#We cover only simpler cases here, but you can set up GridSearchCV and RandomizedSearchCV to tune over any hyperparameters you can think of: 
#optimizers, number of hidden layers, number of neurons per layer, and so on.
###########################################################################################################################################################

from model import design_model, features_train, labels_train

#------------- GRID SEARCH --------------
def do_grid_search():
  #Change the batch_size array to [6, 64] to try other batch size values
  batch_size = [6, 64]
  epochs = [10, 50]
  model = KerasRegressor(build_fn=design_model)
  param_grid = dict(batch_size=batch_size, epochs=epochs)
  grid = GridSearchCV(estimator = model, param_grid=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False),return_train_score = True)
  grid_result = grid.fit(features_train, labels_train, verbose = 0)
  print(grid_result)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

  print("Traininig")
  means = grid_result.cv_results_['mean_train_score']
  stds = grid_result.cv_results_['std_train_score']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

#------------- RANDOMIZED SEARCH --------------
def do_randomized_search():
  param_grid = {'batch_size': sp_randint(2, 16), 'nb_epoch': sp_randint(10, 100)}
  model = KerasRegressor(build_fn=design_model)
  grid = RandomizedSearchCV(estimator = model, param_distributions=param_grid, scoring = make_scorer(mean_squared_error, greater_is_better=False), n_iter = 12)
  grid_result = grid.fit(features_train, labels_train, verbose = 0)
  print(grid_result)
  print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

  means = grid_result.cv_results_['mean_test_score']
  stds = grid_result.cv_results_['std_test_score']
  params = grid_result.cv_results_['params']
  for mean, stdev, param in zip(means, stds, params):
      print("%f (%f) with: %r" % (mean, stdev, param))

print("-------------- GRID SEARCH --------------------")
do_grid_search()
print("-------------- RANDOMIZED SEARCH --------------------")
do_randomized_search()


##########################################################################################################################################################
#Regularization: dropout
#Regularization is a set of techniques that prevent the learning process to completely fit the model to the training data which can lead to overfitting. 
#It makes the model simpler, smooths out the learning curve, and hence makes it more ‘regular’. 
#There are many techniques for regularization such as simplifying the model, adding weight regularization, weight decay, and so on. 
#The most common regularization method is dropout.
#Dropout is a technique that randomly ignores, or “drops out” a number of outputs of a layer by setting them to zeros. 
#The dropout rate is the percentage of layer outputs set to zero (usually between 20% to 50%).
#In Keras, we can add a dropout layer by introducing the Dropout layer.
#Let’s recreate our overfitting network having too many layers and too many neurons per layer in the design_model_no_dropout() method in insurance_tuning.py. 
#For this model, we get the learning curve depicted in Figure 1. The validation error gets worse, which indicates the trend of overfitting.
#Next, we introduce dropout layers in the design_model_dropout() method in insurance_tuning.py. Our model looks like this

##model.add(input)
##model.add(layers.Dense(128, activation = 'relu'))
##model.add(layers.Dropout(0.1))
##model.add(layers.Dense(64, activation = 'relu'))
##model.add(layers.Dropout(0.2))
##model.add(layers.Dense(24, activation = 'relu'))
#your code here!
##model.add(layers.Dense(1))

#For this model, we get the learning curve in Figure 2. The validation MAE we get with the dropout is lower than without it. 
#Note that the validation error is also lower than the training error in this case. One of the explanations might be that the dropout is used only 
#during training, and the full network is used during the validation/testing with layers’ output scaled down by the dropout rate.

from model import features_train, labels_train, fit_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from plotting import plot
import matplotlib.pyplot as plt

def design_model_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(24, activation='relu'))
    #add another dropout method as an instance of tensorflow.keras.layers.Dropout with the dropout rate set to 0.3.
    model.add(layers.Dropout(0.3))


    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def design_model_no_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = layers.InputLayer(input_shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model
    
#using the early stopping in fit_model
learning_rate = 0.001
num_epochs = 200
#train the model without dropout
history1 = fit_model(design_model_no_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
#train the model with dropout
history2 = fit_model(design_model_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)

plot(history1, 'static/images/no_dropout.png')

plot(history2, 'static/images/with_dropout.png')

import app #don't worry about this. This is to show you the plot in the browser.


############################################################################################################################################################
#Baselines: how to know the performance is reasonable?
#Why do we need a baseline? For example, we have data consisting of 90% dog images, and 10% cat images. 
#An algorithm that predicts the majority class for each data point, will have 90% accuracy on this dataset! 
#That might sound good, but predicting the majority class is hardly a useful classifier. We need to perform better.
#A baseline result is the simplest possible prediction. For some problems, this may be a random result, and for others, 
#it may be the most common class prediction. Since we are focused on a regression task in this lesson, 
#we can use averages or medians of the class distribution known as central tendency measures as the result for all predictions.
#Scikit-learn provides DummyRegressor, which serves as a baseline regression algorithm. We’ll choose mean (average) as our central tendency measure.

##from sklearn.dummy import DummyRegressor
##from sklearn.metrics import mean_absolute_error
##dummy_regr = DummyRegressor(strategy="mean")
##dummy_regr.fit(features_train, labels_train)
##y_pred = dummy_regr.predict(features_test)
##MAE_baseline = mean_absolute_error(labels_test, y_pred)
##print(MAE_baseline)

#The result of the baseline is $9,190, and we definitely did better than this (around $3,000) in our previous experiments in this lesson.
##############################################################################################################################################################

from model import features_train, labels_train, features_test, labels_test
import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error

#The result of the baseline is $9,190, and we definitely did better than this (around $3,000) in our previous experiments in this lesson.
dummy_regr = DummyRegressor(strategy="median")
dummy_regr.fit(features_train, labels_train)
y_pred = dummy_regr.predict(features_test)
MAE_baseline = mean_absolute_error(labels_test, y_pred)
print(MAE_baseline)


####################################################################################################################################################
#Summary: 
#In this lesson, you learned how to both manually and automatically choose hyperparameters of the neural network training procedure in order to 
#select a model with the best predictive performance on a validation set. The hyperparameters we covered in this lesson are
#learning rate
#batch size
#number of epochs
#model size (number of hidden layers/neurons and number of parameters)
#regularization (dropout)
#We discussed the concepts of underfitting (having a too simple model to capture data patterns) and overfitting (having a model with too 
#many parameters that learns the training data too well and is not unable to generalize). We discussed methods to combat overfitting such as regularization. 
#To avoid underfitting we increased the complexity of our model.
#Besides data preprocessing, hyperparameter tuning is probably the most costly and intensive process of neural network training. 
#We covered how to set up grid seach and randomized search in Keras in order to automate the process of hyperparameter tuning.
#We also showed you how to check the performance of your model against a simple baseline. 
#Baselines give you an idea of whether your model has a reasonable performance.
#In the end, we hope you see how all of the hyperparameters interplay and how they can influence the performance of the network.
############################################################################################################################################################

rom model import features_train, labels_train, fit_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from plotting import plot
import matplotlib.pyplot as plt

def design_model_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = tf.keras.Input(shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(24, activation='relu'))
    #add another dropout method as an instance of tensorflow.keras.layers.Dropout with the dropout rate set to 0.3.
    model.add(layers.Dropout(0.3))


    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model

def design_model_no_dropout(X, learning_rate):
    model = Sequential(name="my_first_model")
    input = layers.InputLayer(input_shape=(X.shape[1],))
    model.add(input)
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    model.compile(loss='mse', metrics=['mae'], optimizer=opt)
    return model
    
#using the early stopping in fit_model
learning_rate = 0.001
num_epochs = 200
#train the model without dropout
history1 = fit_model(design_model_no_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)
#train the model with dropout
history2 = fit_model(design_model_dropout(features_train, learning_rate), features_train, labels_train, learning_rate, num_epochs)

plot(history1, 'static/images/no_dropout.png')

plot(history2, 'static/images/with_dropout.png')

import app #don't worry about this. This is to show you the plot in the browser.












