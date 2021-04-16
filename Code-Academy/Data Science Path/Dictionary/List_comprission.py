#You have two lists, representing some drinks sold at a coffee shop and the milligrams of caffeine in each. 
#First, create a variable called zipped_drinks that is a zipped list of pairs between the drinks list and the caffeine list.
#Create a dictionary called drinks_to_caffeine by using a list comprehension that goes through the zipped_drinks list and turns each pair into a key:value item.

drinks = ["espresso", "chai", "decaf", "drip"]
caffeine = [64, 40, 0, 120]

zipped_drinks = zip(drinks, caffeine)
drinks_to_caffeine = {key:value for key, value in zipped_drinks}
