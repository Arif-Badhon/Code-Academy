class Perceptron:
  def __init__(self, num_inputs=2, weights=[1,1]):
    self.num_inputs = num_inputs
    self.weights = weights
    
  def weighted_sum(self, inputs):
    weighted_sum = 0
    for i in range(self.num_inputs):
      weighted_sum += self.weights[i]*inputs[i]
    return weighted_sum
  
  def activation(self, weighted_sum):
    if weighted_sum >= 0:
      return 1
    if weighted_sum < 0:
      return -1
    
  def training(self, training_set):
    #First, we need the perceptron’s predicted output for a point. Inside the for loop, create a variable called prediction 
    #and assign it the correct label value using .activation(), .weighted_sum(), and inputs in a single statement
    for inputs in training_set:                   
      prediction = self.activation(self.weighted_sum(inputs))
      #Create a variable named actual and assign it the actual label for each inputs in training_set.
      actual = training_set[inputs]
      #Create a variable called error and assign it the value of actual - prediction.
      error = actual - prediction
      
cool_perceptron = Perceptron()
print(cool_perceptron.weighted_sum([24, 55]))
print(cool_perceptron.activation(52))
