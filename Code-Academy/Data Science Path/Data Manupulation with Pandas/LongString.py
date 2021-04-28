#Create a lambda function named long_string that takes an input str and returns 
#True if the string has over 12 characters in it. Otherwise, return False.

#Write your lambda function here
long_string = lambda str: len(str)>12

print long_string("short")
print long_string("photosynthesis")
