import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')
#they want to know the most expensive shoe for each shoe_type (i.e., the most expensive boot, the most expensive ballet flat, etc.).
pricey_shoes = orders.groupby('shoe_type').price.max()
print(pricey_shoes)
print(type(pricey_shoes))
