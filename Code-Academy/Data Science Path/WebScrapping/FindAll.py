import re
import requests
from bs4 import BeautifulSoup

webpage_response = requests.get('https://content.codecademy.com/courses/beautifulsoup/shellter.html')

webpage = webpage_response.content
soup = BeautifulSoup(webpage, "html.parser")

#Find all of the a elements on the page and store them in a list called turtle_links

turtle_links = soup.find_all(['a'])
print(turtle_links)
