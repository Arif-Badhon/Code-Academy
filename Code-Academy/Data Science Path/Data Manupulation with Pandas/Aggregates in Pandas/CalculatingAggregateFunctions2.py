import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')

#Modify your code from the previous exercise so that it ends with reset_index, which will change pricey_shoes into a DataFrame.
pricey_shoes = orders.groupby('shoe_type').price.max().reset_index()
print(pricey_shoes)
print(type(pricey_shoes))
