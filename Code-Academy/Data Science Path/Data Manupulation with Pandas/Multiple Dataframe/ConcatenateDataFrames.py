import codecademylib
import pandas as pd

bakery = pd.read_csv('bakery.csv')
print(bakery)
ice_cream = pd.read_csv('ice_cream.csv')
print(ice_cream)

#Create their new menu by concatenating the two DataFrames into a DataFrame called menu
menu = pd.concat([bakery, ice_cream])
print(menu)
