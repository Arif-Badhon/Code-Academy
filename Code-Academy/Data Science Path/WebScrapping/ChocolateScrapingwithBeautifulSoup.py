import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Let’s make a request to this site to get the raw HTML,
# which we can later turn into a BeautifulSoup object.
# You can pass this into the .get() method of the requests module to get the HTML.
url = requests.get(
    "https://s3.amazonaws.com/codecademy-content/courses/beautifulsoup/cacao/index.html")

# Create a BeautifulSoup object called soup to traverse this HTML.
# Use "html.parser" as the parser, and the content of the response you got from your request as the document.
soup = BeautifulSoup(url.content, "html.parser")

# If you want, print out the soup object to explore the HTML.
# print(soup)

# How many terrible chocolate bars are out there? And how many earned a perfect 5? Let’s make a histogram of this data.
# The first thing to do is to put all of the ratings into a list.
# Use a command on the soup object to get all of the tags that contain the ratings.
ratings_tags = soup.find_all(attrs={"class": "Rating"})

# Create an empty list called ratings to store all the ratings in.
ratings = []

# Loop through the ratings tags and get the text contained in each one. Add it to the ratings list.
# As you do this, convert the rating to a float, so that the ratings list will be numerical.
# This should help with calculations later.
for text in ratings_tags[1:]:
    ratings.append(float(text.get_text()))

# Using Matplotlib, create a histogram of the ratings values:
plt.hist(ratings)
plt.show()

# We want to now find the 10 most highly rated chocolatiers.
# One way to do this is to make a DataFrame that has the chocolate companies in one column, and the ratings in another.
# Then, we can do a groupby to find the ones with the highest average rating.
# First, let’s find all the tags on the webpage that contain the company names.
company_names_tags = soup.find_all(attrs={"class": "Company"})

# Just like we did with ratings, we now want to make an empty list to hold company names.
company = []

# Loop through the tags containing company names, and add the text from each tag to the list you just created.
for name in company_names_tags[1:]:
    company.append(name.get_text())

# Create a DataFrame with a column “Company” corresponding to your companies list,
# and a column “Ratings” corresponding to your ratings list.
df = pd.DataFrame.from_dict({"Company": company, "Rating": ratings})

# Use .groupby to group your DataFrame by Company and take the average of the grouped ratings.
average_group_rating = df.groupby("Company").Rating.mean()

# Then, use the .nlargest command to get the 10 highest rated chocolate companies. Print them out.
ten_best_rated = average_group_rating.nlargest(10)

print(ten_best_rated)
