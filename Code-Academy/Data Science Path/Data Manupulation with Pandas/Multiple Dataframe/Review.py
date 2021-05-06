import codecademylib
import pandas as pd

visits = pd.read_csv('visits.csv',
                        parse_dates=[1])
checkouts = pd.read_csv('checkouts.csv',
                        parse_dates=[1])

#Use print to inspect each DataFrame.
print(visits)
print(checkouts)

#Use merge to combine visits and checkouts and save it to the variable v_to_c
v_to_c = pd.merge(visits, checkouts)

#In order to calculate the time between visiting and checking out, define a column of v_to_c called time
v_to_c['time'] = v_to_c.checkout_time - \
                 v_to_c.visit_time
 
print(v_to_c)

#To get the average time to checkout
print(v_to_c.time.mean())
