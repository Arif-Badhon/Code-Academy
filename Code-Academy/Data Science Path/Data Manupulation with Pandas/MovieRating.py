#Create a lambda function named rate_movie that takes a number named rating. 
#If rating is greater than 8.5, return "I liked this movie". Otherwise return "This movie was not very good".

#Write your lambda function here
rate_movie = lambda rating: "I liked this movie" if rating > 8.5 else "This movie was not very good"

print rate_movie(9.2)
print rate_movie(7.2)
