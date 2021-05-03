#let’s rename name to movie_title.

#Use the keyword inplace=True so that you modify df rather than creating a new DataFrame!

import codecademylib
import pandas as pd

df = pd.read_csv('imdb.csv')

# Rename columns here
df.rename(columns={
'name': 'movie_title'  
},
inplace = 'True'
)
print(df)
