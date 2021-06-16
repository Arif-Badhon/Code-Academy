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
    foundLine = False
    while not foundLine:
      total_error = 0
      for inputs in training_set:
        prediction = self.activation(self.weighted_sum(inputs))
        actual = training_set[inputs]
        error = actual - prediction
        total_error += abs(error)
        #In order to update the weight for each inputs, create another for loop (inside the existing for loop) that 
        #iterates a loop variable i through a range of self.num_inputs.
        #Inside the second for loop, update each weight self.weights[i] by applying the update rule:
        #weight = weight + (error * inputs)weight=weight+(error∗inputs)
        for i in range(self.num_inputs):
          self.weights[i] += error*inputs[i]
      #If the algorithm doesn’t find an error, the perceptron must have correctly predicted the labels for all points.
      #Outside the for loop (but inside the while loop), change the value of foundLine to True if total_error equals 0.    
      if total_error == 0:
        foundLine = True  
      
cool_perceptron = Perceptron()
small_training_set = {(0,3):1, (3,0):-1, (0,-3):-1, (-3,0):1}
#Great job! Now give it a try for yourself.Train cool_perceptron using small_training_set.
#You can also print out the optimal weights to see for yourself!

cool_perceptron.training(small_training_set)
print(cool_perceptron.weights)
