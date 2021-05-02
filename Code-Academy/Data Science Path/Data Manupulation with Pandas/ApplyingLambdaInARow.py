import codecademylib
import pandas as pd

df = pd.read_csv('employees.csv')


#if an employee worked for 43 hours and made $10/hour, 
#she would receive $400 for the first 40 hours that she worked, and an additional $45 for the 3 hours of overtime, for a total for $445.

total_earned = lambda row: (row.hourly_wage * 40) + ((row.hourly_wage * 1.5) * (row.hours_worked - 40)) \
	if row.hours_worked > 40 \
  else row.hourly_wage * row.hours_worked
  
df['total_earned'] = df.apply(total_earned, axis = 1)

print(df)
