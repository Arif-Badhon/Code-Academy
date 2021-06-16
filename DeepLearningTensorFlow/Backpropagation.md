This all seems fine and dandy so far. However, what if our output values are inaccurate? Do we cry? Try harder next time? Well, we can do that, 
but the good news is that there is more to our deep learning models.

This is where backpropagation and gradient descent come into play. Forward propagation deals with feeding the input values through 
hidden layers to the final output layer. Backpropagation refers to the computation of gradients with an algorithm known as gradient descent. 
This algorithm continuously updates and refines the weights between neurons to minimize our loss function.

By gradient, we mean the rate of change with respect to the parameters of our loss function. From this, 
backpropagation determines how much each weight is contributing to the error in our loss function, and gradient descent will update 
our weight values accordingly to decrease this error.

This is a conceptual overview of backpropagation. If you would like to engage with the gritty mathematics of it, 
you can do so here. However, for this course, we will not need this level of detailed understanding.

Instructions
Let’s take a look at what happens with backpropagation and gradient descent on a neural network directly. 
In the applet in the learning environment, watch as weights are updated and error is decreased after each iteration. 
Without backpropagation, neural networks would be much less accurate.
