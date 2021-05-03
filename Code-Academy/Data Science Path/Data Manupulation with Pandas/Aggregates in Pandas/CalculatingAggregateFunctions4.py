import codecademylib
import numpy as np
import pandas as pd

orders = pd.read_csv('orders.csv')
#Create a DataFrame with the total number of shoes of each shoe_type/shoe_color combination purchased. Save it to the variable shoe_counts.

shoe_counts = orders.groupby(['shoe_color', 'shoe_type']).id\
      .count().reset_index()
print(shoe_counts)
