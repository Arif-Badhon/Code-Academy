mport codecademylib3_seaborn
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

#Create a variable named data that is a list that contains the four possible inputs to an AND gate.

data = [[0,0],[0,1],[1,0],[1,1]]
#Create a variable named labels. This should be a list where each label corresponds to a point in data. For example, if the last item in data is [1, 1], the last label should be 1.
labels = [0,0,0,1]

plt.scatter([point[0] for point in data],[point[1] for point in data],c=labels)
#Build perceptron

classifier = Perceptron(max_iter = 40)
#train model
classifier.fit(data, labels)
#test model
print(classifier.score(data, labels))
#Visualizing the perception
print(classifier.decision_function([[0,0], [1,1], [0.5,0.5]]))

#make a heat map that reveals the decision boundary
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

#now want to find every possible combination of those x and y values.
point_grid = list(product(x_values, y_values))

distances = classifier.decision_function(point_grid)
#Right now distances stores positive and negative values. We only care about how far away a point is from the boundary — we don’t care about the sign
abs_distance = [abs(points) for points in distances]
distances_matrix = np.reshape(abs_distance, (100,100))

#Draw the heat map
heat_map = plt.pcolormesh(x_values, y_values, distances_matrix)
plt.colorbar(heat_map)


plt.show()
