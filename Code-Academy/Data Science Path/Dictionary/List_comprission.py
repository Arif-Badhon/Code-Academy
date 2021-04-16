#You have two lists, representing some drinks sold at a coffee shop and the milligrams of caffeine in each. 
#First, create a variable called zipped_drinks that is a zipped list of pairs between the drinks list and the caffeine list.
#Create a dictionary called drinks_to_caffeine by using a list comprehension that goes through the zipped_drinks list and turns each pair into a key:value item.

drinks = ["espresso", "chai", "decaf", "drip"]
caffeine = [64, 40, 0, 120]

zipped_drinks = zip(drinks, caffeine)
drinks_to_caffeine = {key:value for key, value in zipped_drinks}


#How to create a dictionary
#How to add elements to a dictionary
#How to update elements in a dictionary
#How to use a list comprehension to create a dictionary from two lists
#Let’s practice these skills!

songs = ["Like a Rolling Stone", "Satisfaction", "Imagine", "What's Going On", "Respect", "Good Vibrations"]
playcounts = [78, 29, 44, 21, 89, 5]

plays = {key:value for key, value in zip(songs, playcounts)}
print(plays)

plays.update({"Purple Haze": 1})
plays.update({"Respect": 94})

library = {"The Best Songs": plays, "Sunday Feelings": {}}
print(library)

