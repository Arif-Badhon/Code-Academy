import codecademylib3_seaborn
import pandas as pd
import glob

#First, create a variable called student_files and set it equal to the glob() of all of the csv files we want to import.

student_files = glob.glob("exams*.csv")

#Create an empty list called df_list that will store all of the DataFrames we make from the files exams0.csv through exams9.csv.

df_list = []

#Loop through the filenames in student_files, and create a DataFrame from each file. Append this DataFrame to df_list

for filename in student_files:
  df_list.append(pd.read_csv(filename))
  
#Concatenate all of the DataFrames in df_list into one DataFrame called students.

students = pd.concat(df_list)

print(students.head())
print(len(students))
