import codecademylib
import pandas as pd

store_a = pd.read_csv('store_a.csv')
print(store_a)
store_b = pd.read_csv('store_b.csv')
print(store_b)

#Store A wants to find out what products they carry that Store B does not carry. Using a left merge, combine store_a to store_b and save the results to store_a_b_left.
store_a_b_left = pd.merge(store_a, store_b, how='left')

#Now, Store B wants to find out what products they carry that Store A does not carry. Use a left join, 
#to combine the two DataFrames but in the reverse order (i.e., store_b followed by store_a) and save the results to the variable store_b_a_left

store_b_a_left = pd.merge(store_a, store_b, how='right')

print(store_a_b_left)
print(store_b_a_left)
