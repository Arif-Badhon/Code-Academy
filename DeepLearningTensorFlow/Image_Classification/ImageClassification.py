#######################################............................Prepocessing Image Data..................##########################################

#Our goal is to pass these X-ray images into our network, and to classify them according to their respective labels. At a high-level, 
#this is very similar to our approach for classifying non-image data.
#Now, our features are going to come from image pixels. Each image will be 256 pixels tall and 256 pixels wide, and each pixel has a value between 0 (black) - 255 (white).
#While loading and preprocessing image data can be a bit tricky, Keras provides us with a few tools to make the process less burdensome. Of these, 
#the ImageDataGenerator class is the most critical. We can use ImageDataGenerators to load images from a file path, and to preprocess them. 
#We can constuctor an ImageDataGenerator using the following code

##my_image_data_generator = ImageDataGenerator()

#Beyond just loading our images, the ImageDataGenerator can also preprocess our data. We do this by passing additional arguments to the constructor.
#There are a few ways to preprocess image data, but we will focus on the most important step: pixel normalization. 
#Because neural networks struggle with large integer values, we want to rescale our raw pixel values between 0 and 1. Our pixels have values in [0,255], 
#so we can normalize pixels by dividing each pixel by 255.0.
#We can also use our ImageDataGenerator for data augmentation: generating more data without collecting any new images. 
#A common way to augment image data is to flip or randomly shift each image by small amounts. Because our dataset is only a few hundred images, 
#we’ll also use the ImageDataGenerator to randomly shift images during training.
#For example, we can define another ImageDataGenerator and set its vertical_flip parameter to be True

##my_augmented_image_data_generator = ImageDataGenerator( vertical_flip = True )

#If we use this ImageDataGenerator to load images, it will randomly flip some of those images upside down.
######################################################################################################################################################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Creates an ImageDataGenerator:
training_data_generator = ImageDataGenerator(rescale = 1.0/255.0, zoom_range = 0.2, rotation_range = 15, width_shift_range = 0.05, height_shift_range = 0.05)

#Prints its attributes:
print(training_data_generator.__dict__)


############################.............................Loading Image Data........................................###################################
#Now, we can use the ImageDataGenerator object that we just created to load and batch our data, using its .flow_from_directory() method.
#Let’s consider each of its arguments:
#directory : A string that defines the path to the folder containing our training data.
#class_mode : How we should represent the labels in our data. “For example, we can set this to "categorical" to return our labels as one-hot arrays, 
#with a 1 in the correct class slot.
#color_mode : Specifies the type of image. For example, we set this to "grayscale" for black and white images, or to "rgb" (Red-Green-Blue) for color images.
#target_size : A tuple specifying the height and width of our image. Every image in the directory will be resized into this shape.
#batch_size : The batch size of our data.
#The resulting training_iterator variable is a DirectoryIterator object. We can pass this object directly to model.fit() to train our model on our training data.
#For example, the following code creates a DirectoryIterator that loads images from "my_data_directory" as 128 by 128 pixel color images in batches of 32

##training_data_generator.flow_from_directory(
##"my_data_directory",
##class_mode="categorical",
##color_mode="rgb,
##target_size=(128,128),
##batch_size=32)

#As the training_data_generator loads the data from the directory, it will automatically rescale the pixels by 1/255, and apply the random transformations 
#that we specified in the previous exercise.
#######################################################################################################################################################################

from preprocess import training_data_generator

DIRECTORY = "data/train"
CLASS_MODE = "categorical"
COLOR_MODE = "grayscale"
TARGET_SIZE = (256,256)
BATCH_SIZE = 32

#Creates a DirectoryIterator object using the above parameters:

training_iterator = training_data_generator.flow_from_directory(DIRECTORY,class_mode=CLASS_MODE,color_mode=COLOR_MODE,target_size=TARGET_SIZE,batch_size=BATCH_SIZE)


#########################................................Modifying our Feed-Forward Classification Model...............................####################################
#We will now attempt to adapt a basic feed-forward classification model to classify images
#One way to classify image data is to treat an image as a vector of pixels. After all, we pass most data into our neural networks as feature vectors, 
#so why not do the same here?
#Change the shape of our input layer model to accept our image data. Now, our input shape will be (image height, image width, image channels). 
#For example, if our data were composed of 512x512 pixel RGB images, we add an input shape as follows

##model.add(tf.keras.Input(shape=(512,512,3)))

#Add a Flatten() layer to “flatten” our input image into a single vector. Kera’s Flatten() layer allows us to preserve the batch size of data, 
#but combine the other dimensions of the image (height, width, image channels) into a single, lengthy feature vector. We can then pass this output to a Dense() layer.
############################################################################################################################################################################

import tensorflow as tf

model = tf.keras.Sequential()

#Add an input layer that will expect grayscale input images of size 256x256:

model.add(tf.keras.Input(shape=(256,256,2)))

#Use a Flatten() layer to flatten the image into a single vector:
model.add(tf.keras.layers.Flatten())
#model.add(...)

model.add(tf.keras.layers.Dense(100,activation="relu"))
model.add(tf.keras.layers.Dense(50,activation="relu"))
model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary() 

######################......................................A Better Alternative: Convolutional Neural Networks.....................################################
#Convolutional Neural Networks (CNNs) use layers specifically designed for image data. These layers capture local relationships between nearby features in an image.
#Previously, in our feed-forward model, we multiplied our normalized pixels by a large weight matrix (of shape (65536, 100)) to generate our next set of features.
#However, when we use a convolutional layer, we learn a set of smaller weight tensors, called filters (also known as kernels). We move each of these 
#filters (i.e. convolve them) across the height and width of our input, to generate a new “image” of features. Each new “pixel” results from applying the 
#filter to that location in the original image.
#The interactive on the right demonstrates how we convolve a filter across a single image.
#Why do convolution-based approaches work well for image data?
#Convolution can reduce the size of an input image using only a few parameters.
#Filters compute new features by only combining features that are near each other in the image. 
#This operation encourages the model to look for local patterns (e.g., edges and objects).
#Convolutional layers will produce similar outputs even when the objects in an image are translated (For example, 
#if there were a giraffe in the bottom or top of the frame). 
#This is because the same filters are applied across the entire image.
#Before deep nets, researchers in computer vision would hand design these filters to capture specific information. 
#For example, a 3x3 filter could be hard-coded to activate when convolved over pixels along a vertical or horizontal edge
#However, with deep nets, we can learn each weight in our filter (along with a bias term)! As in our feed-forward layers, 
#these weights are learnable parameters. Typically, we randomly initialize our filters and use gradient descent to learn a better set of weights. 
#By randomly initializing our filters, we ensure that different filters learn different types of information, like vertical versus horizontal edge detectors
########################################################################################################################################################################

##############################..............................Configuring a Convolutional Layer - Filters..............................#######################################
#In Keras, we can define a Conv2D layer to handle the forward and backward passes of convolution
##Defines a convolutional layer with 4 filters, each of size 5 by 5:
 
##tf.keras.layers.Conv2D(4, 5, activation="relu"))  

#When defining a convolutional layer, we can specify the number and size of the filters that we convolve across each image.
#Number of Filters
#When using convolutional layers, we don’t just convolve one filter. Instead, we define some number of filters. 
#We convolve each of these in turn to produce a new set of features. Then we stack these outputs (one for each filter) together in a new “image.”
#Our output tensor is then (batch_size, new height, new width, number of filters). We call this last dimension number of channels ( or feature maps ). 
#These are the result of applying a single filter across the entire image.
#Filter Size
#Beyond configuring the number of filters, we can also configure their size. Each filter has three dimensions: [Height, Width, Input Channels]
#Height: the height of our filter (in pixels)
#Width: the width of our filter (also in pixels)
#Input Channels: The number of input channels. In a black and white image, there is 1 input channel (grayscale). However, 
#in an RGB image, there are three input channels. Note that we don’t have control over this dimension (it depends on the input), 
#and Keras takes care of this last dimension for us.
#Increasing height or width increases the number of pixels that a filter can pay attention to at each step in the convolution. However, 
#doing so also increases the number of learnable parameters. People commonly use filters of size 5x5 and 3x3.
#In total, the number of parameters in a convolution layer is:

##Number of filters×(Input Channels×Height×Width+1)

#Every filter has height, width, and thickness (The number of input channels), along with a bias term.
########################################################################################################################################################################

import tensorflow as tf

#This convolutional layer has 8 filters, and each is 3x3.

#Observe the dimensionality of the output, and the number of parameters.
print("\n\nModel with 8 filters:")
 
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))
 
#Adds a Conv2D layer with 8/16 filters, each size 3x3/7x7:
model.add(tf.keras.layers.Conv2D(16, 7,activation="relu"))
model.summary()

##################################................Configuring a Convolutional Layer - Stride and Padding....................###############################################
#Two other hyperparameters in a convolutional layer are Stride and Padding.
#Stride
#The stride hyperparameter is how much we move the filter each time we apply it. The default stride is 1, meaning that we move the filter across the image 1-pixel at a time. 
#When we reach the end of a row in the image, we then go to the next one.
#If we use a stride greater than 1, we do not apply our filter centered on every pixel. Instead, we move the filter multiple pixels at a time.
#For example, if strides = 2, we move the filter two columns over at a time, and then skip every other row.
#We can set the stride to any integer. For example, we can define a Conv2D layer with a stride of 3
#Adds a Conv2D layer with 8 filters, each size 5x5, and uses a stride of 3:

##model.add(tf.keras.layers.Conv2D(8, 5,
##strides=3,
##activation="relu"))

#Larger strides allow us to decrease the size of our output. In the case where our stride=2, we apply our filter to every other pixel. 
#As a result, we will halve the height and width of our output.

#Padding
#The padding hyperparameter defines what we do once our filter gets to the end of a row/column. In other words: “what happens when we run out of image?” 
#There are two main methods for what to do here
#We just stop (valid padding): The default option is to just stop when our kernel moves off the image. 
#Let’s say we are convolving a 3x3 filter across a 7x7 image with stride=1. Our output will then be a 5x5 image, 
#because we can’t process the 6th pixel without our filter hanging off the image.
#We keep going (same padding): Another option is to pad our input by surrounding our input with zeros. In this case, 
#if we add zeros around our 7x7 image, then we can apply the 3x3 filter to every single pixel. This approach is called “same” padding, 
#because if stride=1, the output of the layer will be the same height and width as the input.
#We can use “same” padding by setting the padding parameter

##Adds a Conv2D layer with 8 filters, each size 5x5, and uses a stride of 3:
##model.add(tf.keras.layers.Conv2D(8, 5,
##strides=3,
##padding='same',
##activation="relu"))
##############################################################################################################################################################################

import tensorflow as tf

print("Model with 16 filters:")

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))

#Adds a Conv2D layer with 16 filters, each size 7x7, and uses a stride of 1 with valid padding:
#Change the number of strides to 2.
#Change the padding type to be from "valid" to "same", and set strides equal to 1
model.add(tf.keras.layers.Conv2D(16, 7,
strides=1,
padding="same",
activation="relu"))
model.summary()

################............OUTPUT:...................
#Model with 16 filters:
#Model: "sequential"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d (Conv2D)              (None, 256, 256, 16)      800       
#=================================================================
#Total params: 800
#Trainable params: 800
#Non-trainable params: 0
#_______________________________

##############################......................Adding Convolutional Layers to Your Model....................######################################################
#Adding One Convolutional Layer
#Now, we can modify our feed-forward image classification code to use a convolutional layer
#First, we are going to replace the first two Dense layers with a Conv2D layer.
#Then, we want to move the Flatten layer between the convolutional and last dense layer. 
#Because dense layers apply their matrix to the dimension, we will always need to flatten the output of convolutional layers before passing them into a dense layer.

#Stacking Convolutional Layers
#However, we won’t stop there! The beauty of neural networks is that we can stack many layers to learn richer combinations of features. 
#We can stack convolutional layers the same way we stacked dense layers.
#For example, we can stack three convolutional layers with distinct filter shapes and strides:

# 8 5x5 filters, with strides of 3
##model.add(tf.keras.layers.Conv2D(8, 5, strides=3, activation="relu"))
 
# 4 3x3 filters, with strides of 3
##model.add(tf.keras.layers.Conv2D(4, 3, strides=3, activation="relu"))
 
# 2 2x2 filters, with strides of 2
##model.add(tf.keras.layers.Conv2D(2, 3, strides=2, activation="relu"))

#Like with dense layers, the output of one convolutional layer can be passed as input to another. 
#You can think of the output as a new input “image,” with a height, width, and number of channels. 
#The number of filters used in the previous layer becomes the number of channels that we input into the next!
#As with dense layers, we should use non-linear activation functions between these convolutional layers.
##########################################################################################################################################################################

import tensorflow as tf

model = tf.keras.Sequential()


model.add(tf.keras.Input(shape=(256,256,1)))

#Add a Conv2D layer
# - with 2 filters of size 5x5
# - strides of 3
# - valid padding
model.add(tf.keras.layers.Conv2D(2, 5, strides=3, activation="relu"))
model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu"))
model.add(tf.keras.layers.Flatten())

# #Remove these two dense layers:
#model.add(tf.keras.layers.Dense(100,activation="relu"))
#model.add(tf.keras.layers.Dense(50,activation="relu"))

model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary()


################.....................Output..........########

#Model: "sequential"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d (Conv2D)              (None, 84, 84, 2)         52        
#_________________________________________________________________
#conv2d_1 (Conv2D)            (None, 82, 82, 4)         76        
#_________________________________________________________________
#flatten (Flatten)            (None, 26896)             0         
#_________________________________________________________________
#dense (Dense)                (None, 2)                 53794     
#=================================================================
#Total params: 53,922
#Trainable params: 53,922
#Non-trainable params: 0
#______________________________



#####################################...................................Pooling.............................#################################################
#Another part of Convolutional Networks is Pooling Layers: layers that pool local information to reduce the dimensionality of intermediate convolutional outputs.
#There are many different types of pooling layer, but the most common is called Max pooling
#Like in convolution, we move windows of specified size across our input. We can specify the stride and padding in a max pooling layer
#However, instead of multiplying each image patch by a filter, we replace the patch with its maximum value.
#For example, we can define a max pooling layer that will move a 3x3 window across the input, with a stride of 3 and valid padding

##max_pool_2d = tf.keras.layers.MaxPooling2D(pool_size=(3, 3),   strides=(3, 3), padding='valid')

#Beyond helping reduce the size of hidden layers (and reducing overfitting), max pooling layers have another useful property: 
#they provide some amount of translational invariance. In other words, even if we move around objects in the input image, 
#the output will be the same. This is very useful for classification. For example, we usually want to classify an image of a cat as a cat, 
#regardless of how the cat is oriented in the image
##########################################################################################################################################################################

import tensorflow as tf

model = tf.keras.Sequential()


model.add(tf.keras.Input(shape=(256,256,1)))

model.add(tf.keras.layers.Conv2D(2,5,strides=3,padding="valid",activation="relu"))

#Add first max pooling layer here.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5,5)))
#model.add(...)

model.add(tf.keras.layers.Conv2D(4,3,strides=1,padding="valid",activation="relu"))

#Add the second max pooling layer here.
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), strides=(2,2)))
#model.add(...)

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(2,activation="softmax"))

#Print model information:
model.summary() 


#####################.................Output................###################################
#Model: "sequential"
#_________________________________________________________________
#Layer (type)                 Output Shape              Param #   
#=================================================================
#conv2d (Conv2D)              (None, 84, 84, 2)         52        
#_________________________________________________________________
#max_pooling2d (MaxPooling2D) (None, 16, 16, 2)         0         
#_________________________________________________________________
#conv2d_1 (Conv2D)            (None, 14, 14, 4)         76        
#_________________________________________________________________
#max_pooling2d_1 (MaxPooling2 (None, 7, 7, 4)           0         
#_________________________________________________________________
#flatten (Flatten)            (None, 196)               0         
#_________________________________________________________________
#dense (Dense)                (None, 2)                 394       
#=================================================================
#Total params: 522
#Trainable params: 522
#Non-trainable params: 0
#_____________________________________________


################################........................Training the Model.....................#######################
#Now, we are going to put everything together and train our model!
#We have do do three additional things
#Define another ImageDataGenerator and use it to load our validation data.
#Compile our model with an optimizer, metric, and a loss function.
#Train our model using model.fit().
#Validation Data Generator
#We have already defined an ImageDataGenerator called training_data_generator. Like in the second exercise, we use training_data_generator.flow_from_directory() 
#to preprocess and augment our training data.
#Now, we need another ImageDataGenerator to load our validation data, which consists of 100 Normal X-rays, and 100 with Pneumonia. 
#Like with our training data, we are going to need to normalize our pixels. However, unlike for our training data, we will not augment the validation data with random shifts.
#Loss, Optimizer, and Metrics
#Because our labels are onehot ([1,0] and [0,1]), we will use keras.losses.CategoricalCrossentropy. We will optimize this loss using the Adam optimizer.
#Because our dateset is balanced, accuracy is a meaningful metric. We will also include AUC (area under the ROC curve). 
#An ROC curve gives us the relationship between our true positive rate and our false positive rate. 
#A true positive would be correctly identifying a patient with Pneumonia, while a false positive would be incorrectly identifying a healthy person as having pneumonia. 
#Like with accuracy, we want our AUC to be as close to 1.0 as possible.
#Training the Model
#To train our model, we have to call model.fit() on our training data DirectoryIterator and validation data DirectoryIterator.
#To reap the benefits of data augmentation, we will iterate over our training data five times (five epochs).
############################################################################################################################################################################

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

BATCH_SIZE = 16

print("\nLoading training data...")

training_data_generator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05)

training_iterator = training_data_generator.flow_from_directory('data/train',class_mode='categorical',color_mode='grayscale',batch_size=BATCH_SIZE)


print("\nLoading validation data...")

#1) Create validation_data_generator, an ImageDataGenerator that just performs pixel normalization:

validation_data_generator = ImageDataGenerator(rescale = 1.0/255)

#2) Use validation_data_generator.flow_from_directory(...) to load the validation data from the 'data/test' folder:

validation_iterator = validation_data_generator.flow_from_directory('data/test',class_mode='categorical',color_mode='grayscale',batch_size=BATCH_SIZE)


print("\nBuilding model...")

#Rebuilds our model from the previous exercise, with convolutional and max pooling layers:

model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(256, 256, 1)))
model.add(tf.keras.layers.Conv2D(2, 5, strides=3, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(5, 5), strides=(5,5)))
model.add(tf.keras.layers.Conv2D(4, 3, strides=1, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(2,activation="softmax"))

model.summary()


print("\nCompiling model...")

#3) Compile the model with an Adam optimizer, Categorical Cross Entropy Loss, and Accuracy and AUC metrics:

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
)

print("\nTraining model...")

#4) Use model.fit(...) to train and validate our model for 5 epochs:

model.fit(
        training_iterator,
        steps_per_epoch=training_iterator.samples/BATCH_SIZE,
        epochs=5,
        validation_data=validation_iterator,
        validation_steps=validation_iterator.samples)


###############################...............................What do filters learn.....................#################################################
#We just trained our neural network to classify chest X-rays. Now, let’s peek at what our different filters actually learn.
#There are a few ways to visualize the internal workings of our model. When working with convolutional networks, 
#one of the most common approaches is to generate feature maps: the result of convolving a single filter across our input.
#Feature maps allow us to see how our network responds to a particular image in ways that are not always apparent when we only examine the raw filter weights.
#For example, consider this x-ray, which our model correctly classifies as Pneumonia
#################################################################################################################################################################

#################################.................................Review.....................................#################################################
#In this lesson, we have covered:

#How to use Keras to preprocess and load image data.
#How we can use Convolutional Neural Networks to classify images, and why these models outperform feed-forward models based on linear layers.
#How we can adjust the filter dimensions, stride, and padding of a convolutional layer, and how these hyperparameters affect output size.
#How we can use pooling methods to further reduce the size of our hidden layers and also gain some spatial invariance properties.
#Examples of the patterns learned by different convolutional filters.








