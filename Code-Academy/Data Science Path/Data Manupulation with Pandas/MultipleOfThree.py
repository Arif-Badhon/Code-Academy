#Create a lambda function named multiple_of_three that takes an integer named num. If num is a multiple of three, 
#return "multiple of three". Otherwise, return "not a multiple"

#Write your lambda function here
multiple_of_three = lambda num: "multiple of three" if num%3==0 else "not a multiple"
print multiple_of_three(9)
print multiple_of_three(10)
