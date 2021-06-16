We have seen how we get to an output! Now, what do we do with it? When a value is outputted, we calculate its error using a loss function. 
Our predicted values are 
compared with the actual values within the training data. There are two commonly used loss calculation formulas:

Mean squared error, which is most likely familiar to you if you have come across linear regression. This gif below shows 
how mean squared error is calculated for a line of best fit in linear regression.A visual that shows the squared distance of 
each point from a line of best fit. The formula for this is (predicted value - actual value)^2Click here to view the gif in a full-sized page.
Cross-entropy loss, which is used for classification learning models rather than regression.
You will learn more about this as you use loss functions in your deep learning models.

Instructions
The interactive visualization in the browser lets you try to find the line of best fit for a random set of data points:

The slider on the left controls the m (slope)
The slider on the right controls the b (intercept)
You can see the total squared error on the right side of the visualization. To get the line of best fit, we want this loss to be as small as possible.
To check if you got the best line, check the “Plot Best-Fit” box.

Randomize a new set of points and try to fit a new line by entering the number of points you want (try 8!) in the textbox and pressing Randomize Points.

Play around with the interactive applet, and notice what method you use to minimize loss:

Do you first get the slope to where it produces lowest loss, and then move the intercept to where it produces lowest loss?
Do you create a rough idea in your mind where the line should be first, and then enter the parameters to match that image?
