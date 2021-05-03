import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')
#Our finance department wants to know the price of the most expensive pair of shoes purchased. Save your answer to the variable most_expensive
most_expensive = orders.price.max()
#Our fashion department wants to know how many different colors of shoes we are selling. Save your answer to the variable num_colors
num_colors = orders.shoe_color.nunique()
print(orders.head(10))
