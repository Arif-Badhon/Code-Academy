import codecademylib
import pandas as pd

orders = pd.read_csv('orders.csv')

products = pd.read_csv('products.csv')

customers = pd.read_csv('customers.csv')

print(orders)
print(products)
print(customers)

order_3_description = 'thing-a-ma-jig'
order_5_phone_number = '112-358-1321'


import codecademylib
import pandas as pd

sales = pd.read_csv('sales.csv')
print(sales)
targets = pd.read_csv('targets.csv')
print(targets)

#Create a new DataFrame sales_vs_targets which contains the merge of sales and targets.
sales_vs_targets = pd.merge(sales, targets)
print(sales_vs_targets)


#Select the rows from sales_vs_targets where revenue is greater than target. Save these rows to the variable crushing_it.
crushing_it = sales_vs_targets[sales_vs_targets.revenue > sales_vs_targets.target]


import codecademylib
import pandas as pd

sales = pd.read_csv('sales.csv')
print(sales)
targets = pd.read_csv('targets.csv')
print(targets)

#Merge all three DataFrames (sales, targets, and men_women) into one big DataFrame called all_data
men_women = pd.read_csv('men_women_sales.csv')
all_data = sales.merge(targets).merge(men_women)
print(all_data)

#Cool T-Shirts Inc. thinks that they have more revenue in months where they sell more women’s t-shirts
results = all_data[(all_data.revenue > all_data.target) & (all_data.women > all_data.men)]
