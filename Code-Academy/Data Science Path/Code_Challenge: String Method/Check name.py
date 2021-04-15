#Write a function called check_for_name that takes two strings as parameters named sentence and name. 
#The function should return True if name appears in sentence in all lowercase letters, all uppercase letters, or with any mix of uppercase and lowercase letters. 
#The function should return False otherwise.

# Write your check_for_name function here:
def check_for_name(sentence, name):
  name = name.lower()
  sentence = sentence.lower()
  if name in sentence:
    return True
  return False
# Uncomment these function calls to test your  function:
print(check_for_name("My name is Jamie", "Jamie"))
# should print True
print(check_for_name("My name is jamie", "Jamie"))
# should print True
print(check_for_name("My name is Samantha", "Jamie"))
# should print False
