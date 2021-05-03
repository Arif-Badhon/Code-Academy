#Once more, you’ll be the data analyst for ShoeFly.com, a fictional online shoe store.

#More messy order data has been loaded into the variable orders. Examine the first 5 rows of the data using print and head.

import codecademylib
import pandas as pd

orders = pd.read_csv('shoefly.csv')

print(orders.head(5))

#Many of our customers want to buy vegan shoes (shoes made from materials that do not come from animals). 
#Add a new column called shoe_source, which is vegan if the materials is not leather and animal otherwise.

orders['shoe_source'] = orders.shoe_material.apply(lambda x: \
                        	'animal' if x == 'leather'else 'vegan')


#Our marketing department wants to send out an email to each customer. 
#Using the columns last_name and gender create a column called salutation which contains Dear Mr. <last_name> for men and Dear Ms. <last_name> for women.

orders['salutation'] = orders.apply(lambda row: \
                                    'Dear Mr. ' + row['last_name']
                                    if row['gender'] == 'male'
                                    else 'Dear Ms. ' + row['last_name'],
                                    axis=1)
