#Create a lambda function named ends_in_a that takes an input str and returns True 
#if the last character in the string is an a. Otherwise, return False.

#Write your lambda function here
ends_in_a = lambda str: str[-1] == 'a'

print ends_in_a("data")
print ends_in_a("aardvark")
