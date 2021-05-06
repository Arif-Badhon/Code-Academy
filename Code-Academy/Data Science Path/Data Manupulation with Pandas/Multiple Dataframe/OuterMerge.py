import codecademylib
import pandas as pd

store_a = pd.read_csv('store_a.csv')
print(store_a)
store_b = pd.read_csv('store_b.csv')
print(store_b)

#There are two hardware stores in town: Store A and Store B. Store A’s inventory is in DataFrame store_a and Store B’s 
#inventory is in DataFrame store_b. They have decided to merge into one big Super Store!
#Combine the inventories of Store A and Store B using an outer merge. Save the results to the variable store_a_b_outer.

store_a_b_outer = pd.merge(store_a, store_b, how='outer')
print(store_a_b_outer)
