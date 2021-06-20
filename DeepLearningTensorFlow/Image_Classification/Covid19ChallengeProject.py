#Dataset is from: https://www.kaggle.com/pranavraikokte/covid19-image-dataset

###############...........................Covid-19 and Pneumonia Classification with Deep Learning.......................################################
#Images are in grayscale
#This is a multiclass problem with 3 classes: 1.Covid 2.Viral Pneumonia 3.Normal
#Train and Test data are seperated in folders already
##########################################################################################################################################################

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers

import matplotlib.pyplot as plt
import app

#Load the data
#Train Data
training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, zoom_range = 0.2, rotation_range = 15, width_shift_range = 0.05, height_shift_range = 0.05)
DIRECTORY = "augmented-data/train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32
training_iterator = training_data_generator.flow_from_directory(DIRECTORY,class_mode=CLASS_MODE,color_mode=COLOR_MODE,target_size=TARGET_SIZE,batch_size=BATCH_SIZE)

#Validation Data
validation_data_generator = ImageDataGenerator(rescale = 1.0/255)

validation_iterator = validation_data_generator.flow_from_directory('augmented-data/test',class_mode='categorical',color_mode='grayscale',batch_size=BATCH_SIZE)

#Model Building
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))
model.add(tf.keras.layers.Conv2D(2, 5, strides=3, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(5, 5), strides=(5,5)))
model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(3,activation="softmax"))

model.summary()

#compile model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

#Fit the data into model
model.fit(
        training_iterator,
        steps_per_epoch=training_iterator.samples/BATCH_SIZE,
        epochs=5,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples)
