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
    #Complete this method
    #Inside the .activation() method, return 1 if the weighted_sum is greater than or equal to 0.
    #Inside the .activation() method, return -1 if the weighted_sum is less than 0.
    if weighted_sum >= 0:
      return 1
    else:
       return -1 

cool_perceptron = Perceptron()
print(cool_perceptron.weighted_sum([24, 55]))
#Print out the result of the method .activation() called on cool_perceptron if the weighted sum is 52.
print(cool_perceptron.activation(52))
