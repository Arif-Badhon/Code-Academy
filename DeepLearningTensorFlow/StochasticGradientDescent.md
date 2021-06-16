This leads us to the final point about gradient descent. In deep learning models, 
we are often dealing with extremely large datasets. Because of this, performing backpropagation and gradient descent calculations on all 
of our data may be inefficient and computationally exhaustive no matter what learning rate we choose.

To solve this problem, a variation of gradient descent known as Stochastic Gradient Descent (SGD) was developed. 
Let’s say we have 100,000 data points and 5 parameters. If we did 1000 iterations (also known as epochs in Deep Learning) 
we would end up with 100000⋅5⋅1000 = 500,000,000 computations. We do not want our computer to do that many computations on top of the 
rest of the learning model; it will take forever.

This is where SGD comes to play. Instead of performing gradient descent on our entire dataset, 
we pick out a random data point to use at each iteration. This cuts back on computation time immensely while still yielding accurate results.
