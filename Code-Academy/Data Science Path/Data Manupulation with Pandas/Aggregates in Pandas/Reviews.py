import codecademylib
import pandas as pd

user_visits = pd.read_csv('page_visits.csv')

print(user_visits.head())

#The column utm_source contains information about how users got to ShoeFly’s homepage. For instance, 
#if utm_source = Facebook, then the user came to ShoeFly by clicking on an ad on Facebook.com
click_source = user_visits.groupby('utm_source').id.count()\
           .reset_index()

print(click_source)

#Our Marketing department thinks that the traffic to our site has been changing over the past few months. 
#Use groupby to calculate the number of visits to our site from each utm_source for each month.
click_source_by_month = user_visits.groupby(['utm_source', 'month']).id.count().reset_index()

click_source_by_month_pivot = click_source_by_month.pivot(
  columns = 'month',
	index = 'utm_source',
	values = 'id').reset_index()

print(click_source_by_month_pivot)
