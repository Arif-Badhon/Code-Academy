import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')

#In script.py, you’ll find two DataFrames: products and orders. Inspect these DataFrames using print.
print(orders)
print(products)

#Merge orders and products and save it to the variable merged_df.

merged_df = pd.merge(orders, products)

print(merged_df)
