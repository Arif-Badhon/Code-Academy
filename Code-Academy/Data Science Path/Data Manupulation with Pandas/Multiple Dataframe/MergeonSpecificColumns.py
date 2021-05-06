import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')
print(orders)
products = pd.read_csv('products.csv')
print(products)

#Merge orders and products using rename. Save your results to the variable orders_products
orders_products = pd.merge(
  orders,
  products.rename(columns={'id': 'product_id'})
)
print(orders_products)


import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')
products = pd.read_csv('products.csv')

# Use left_on and right_on to merge orders and products
orders_products = pd.merge(
	orders,
	products,
	left_on = 'product_id',
	right_on = 'id',
	suffixes = ['_orders', '_products']
)

print(orders_products)
