import codecademylib3_seaborn
import pandas as pd
from students import students

print(students)

duplicates = students.duplicated()
duplicates.value_counts()

students = students.drop_duplicates()

duplicates = students.duplicated()
