We have the overall process of backpropagation down! Now, let’s zoom in on what is happening during gradient descent.

If we think about the concept graphically, we want to look for the minimum point of our loss function because 
this will yield us the highest accuracy. If we start at a random point on our loss function, gradient descent 
will take “steps” in the “downhill direction” towards the negative gradient. The size of the “step” taken is 
dependent on our learning rate. Choosing the optimal learning rate is important because it affects both the efficiency and accuracy of our results.

The formula used with learning rate to update our weight parameters is the following:

parameter\_new=parameter\_old+learning\_rate \cdot gradient(loss\_function(parameter\_old))parameter_new=parameter_old+learning_rate⋅gradient(loss_function(parameter_old))
The learning rate we choose affects how large the “steps” our pointer takes when trying to optimize our error function. 
Initial intuition might indicate that you should choose a large learning rate; however, as shown above, 
this can lead you to overshoot the value we are looking for and cause a divergent search.

Now you might think that you should choose an incredibly small learning rate; however, if it is too small, 
it could cause your model to be unbearably inefficient or get stuck in a local minimum and never find the optimum value. 
It is a tricky game of finding the correct combination of efficiency and accuracy.
