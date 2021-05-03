import codecademylib
import pandas as pd

ad_clicks = pd.read_csv('ad_clicks.csv')

print(ad_clicks.head(10))

#Your manager wants to know which ad platform is getting you the most views
ad_clicks.groupby('utm_source')\
    .user_id.count()\
    .reset_index()

#If the column ad_click_timestamp is not null, then someone actually clicked on the ad that was displayed
ad_clicks['is_click'] = ~ad_clicks\
   .ad_click_timestamp.isnull()

#We want to know the percent of people who clicked on ads from each utm_source.
clicks_by_source = ad_clicks\
   .groupby(['utm_source',
             'is_click'])\
   .user_id.count()\
   .reset_index()

#pivot the data so that the columns are is_click (either True or False), the index is utm_source, and the values are user_id

clicks_pivot = clicks_by_source.pivot(
  index='utm_source',
  columns='is_click',
  values = 'user_id'
).reset_index()

#Create a new column in clicks_pivot called percent_clicked which is equal to the percent of users who clicked on the ad from each utm_source
clicks_pivot['percent_clicked'] = clicks_pivot[True] / (clicks_pivot[True] + clicks_pivot[False])

#The column experimental_group tells us whether the user was shown Ad A or Ad B.
ad_clicks.groupby('experimental_group').user_id.count().reset_index()

#Using the column is_click that we defined earlier, check to see if a greater percentage of users clicked on Ad A or Ad B.

ad_clicks.groupby(['experimental_group', 'is_click']).user_id.count().reset_index()\
 .pivot(
   index = 'experimental_group',
   columns = 'is_click',
   values = 'user_id'
 ).reset_index()

#The Product Manager for the A/B test thinks that the clicks might have changed by day of the week
a_clicks = ad_clicks[
   ad_clicks.experimental_group
   == 'A']
b_clicks = ad_clicks[
  ad_clicks.experimental_group == 'B'
]
#For each group (a_clicks and b_clicks), calculate the percent of users who clicked on the ad by day.
a_clicks_pivot = a_clicks.groupby(['is_click', 'day']).user_id.count().reset_index()\
 .pivot(
   index = 'day',
   columns = 'is_click',
   values = 'user_id'
 ).reset_index()
a_clicks_pivot['percent_clicked'] = a_clicks_pivot[True] / (a_clicks_pivot[True] + a_clicks_pivot[False])

print(a_clicks_pivot)
##########
b_clicks_pivot = b_clicks.groupby(['is_click', 'day']).user_id.count().reset_index()\
 .pivot(
   index = 'day',
   columns = 'is_click',
   values = 'user_id'
 ).reset_index()
b_clicks_pivot['percent_clicked'] = b_clicks_pivot[True] / (a_clicks_pivot[True] + b_clicks_pivot[False])

print(b_clicks_pivot)

#Compare the results for A and B. What happened over the course of the week
