import codecademylib3_seaborn
import pandas as pd
from students import students

#Use regex to take out the % signs in the score column.
students.score = students['score'].replace('[\%,]', '',regex=True)

#Convert the score column to a numerical type using the pd.to_numeric() function
students.score = pd.to_numeric(students.score)
