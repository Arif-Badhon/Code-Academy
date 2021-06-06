import codecademylib3_seaborn
import pandas as pd
from students import students

#Get the mean of the score column. Store it in score_mean and print it out
score_mean = students['score'].mean()

#Fill all of the nans in students['score'] with 0
students.score = students.score.fillna(0)

#Get the mean of the score column again. Store it in score_mean_2 and print it out
score_mean_2 = students['score'].mean()

print(students)
