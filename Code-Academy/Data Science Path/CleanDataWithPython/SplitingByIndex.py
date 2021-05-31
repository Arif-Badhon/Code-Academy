import codecademylib3_seaborn
import pandas as pd
from students import students

#print(students.columns)
#print(students.head())

students['gender'] = students.gender_age.str[0]
students['age'] = students.gender_age.str[1:]
print(students.head())

students = students[['full_name', 'gender', 'age', 'grade', 'exam', 'score']]
