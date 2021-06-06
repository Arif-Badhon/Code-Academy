import codecademylib3_seaborn
import pandas as pd
from students import students



#Use regex to extract the number from each string in grade and store those values back into the grade column
students['grade'] = students['grade'].str.split('(\d+)', expand=True)[1]

#print grade column
print(students.grade.head())

#Print the dtypes of the students table
print(students.dtypes)

#Convert the grade column to be numerical values instead of objects
students.grade = pd.to_numeric(students.grade)

#Calculate the mean of grade, store it in a variable called avg_grade, and then print it out!
avg_grade = students['grade'].mean()
print(students)
