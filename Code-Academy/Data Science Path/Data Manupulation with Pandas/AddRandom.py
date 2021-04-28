#Create a lambda function named add_random that takes an input named num. 
#The function should return num plus a random integer number between 1 and 10 (inclusive).

import random
#Write your lambda function here
add_random = lambda num: num+random.randint(1,10)
print add_random(5)
print add_random(100)
