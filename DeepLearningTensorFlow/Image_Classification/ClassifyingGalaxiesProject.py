#Data: https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/
#################################.......................Classifying Galaxies Using Convolutional Neural Networks.............###########################

#Around the clock, telescopes affixed to orbital satellites and ground-based observatories are taking millions of pictures 
#of millions upon millions of celestial bodies. These data, of stars, planets and galaxies provide an invaluable resource to astronomers.
#However, there is a bottleneck: until the data is annotated, it’s incredibly difficult for scientists to put it to good use. 
#Additionally, scientists are usually interested in subsets of the data, like galaxies with unique characteristics.
#In this project, you will build a neural network to classify deep-space galaxies. 
#You will be using image data curated by Galaxy Zoo, a crowd-sourced project devoted to annotating galaxies in support of scientific discovery.
#You will identify “odd” properties of galaxies. The data falls into four classes:
#[1,0,0,0] - Galaxies with no identifying characteristics.
#[0,1,0,0] - Galaxies with rings.
#[0,0,1,0] - Galactic mergers.
#[0,0,0,1] - “Other,” Irregular celestial bodies.
##########################################################################################################################################################

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn.model_selection import train_test_split
from utils import load_galaxy_data

import app


input_data, labels = load_galaxy_data()

#Use .shape to print the dimensions of the input_data and labels.
print(input_data.shape)
#(1400, 128, 128, 3)
print(labels.shape)
#(1400, 4)

#split the train and test data
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size = 0.20, stratify=labels, shuffle=True, random_state=222)

#preprocess the input
data_generator = ImageDataGenerator(rescale=1./255)

#Now create two data iterator
training_iterator = data_generator.flow(x_train, y_train,batch_size=5)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=5)

#now build model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128, 128, 3)))
model.add(tf.keras.layers.Dense(4,activation="softmax"))

#Define layers of model
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="softmax"))

#now add optimizer, loss, and metrics
#labels are one hot categories
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

#summary of model
model.summary()

#Fit the model
model.fit(
        training_iterator,
        steps_per_epoch=len(x_train)/5,
        epochs=8,
        validation_data=validation_iterator,
        validation_steps=len(x_valid)/5)
