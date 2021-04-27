#After the loop, print out turtle_data. We have been storing the names as the whole p tag containing the name.

#Instead, let’s call get_text() on the turtle_name element and store the result as the key of our dictionary instead.
#Instead of associating each turtle with an empty list, let’s have each turtle associated with a list of the stats that are available on their page.

#It looks like each piece of information is in a li element on the turtle’s page.

#Get the ul element on the page, and get all of the text in it, separated by a '|' character so that we can easily split out each attribute later.

#Store the resulting string in turtle_data[turtle_name] instead of storing an empty list there.

#When we store the list of info in each turtle_data[turtle_name], separate out each list element again by splitting on '|'.

import requests
from bs4 import BeautifulSoup

prefix = "https://content.codecademy.com/courses/beautifulsoup/"
webpage_response = requests.get('https://content.codecademy.com/courses/beautifulsoup/shellter.html')

webpage = webpage_response.content
soup = BeautifulSoup(webpage, "html.parser")

turtle_links = soup.find_all("a")
links = []
#go through all of the a tags and get the links associated with them"
for a in turtle_links:
  links.append(prefix+a["href"])
    
#Define turtle_data:
turtle_data = {}

#follow each link:
for link in links:
  webpage = requests.get(link)
  turtle = BeautifulSoup(webpage.content, "html.parser")
  turtle_name = turtle.select(".name")[0].get_text()
  
  stats = turtle.find("ul")
  stats_text = stats.get_text("|")
  turtle_data[turtle_name] = stats_text.split("|")
    
