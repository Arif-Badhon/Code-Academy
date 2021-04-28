#Create a lambda function named even_or_odd that takes an integer named num. If num is even, return "even". 
#If num is odd, return "odd".

#Write your lambda function here
even_or_odd = lambda num: "even" if num%2==0 else "odd"

print even_or_odd(10)
print even_or_odd(5)
