# import sha256
from hashlib import sha256
# text to hash
text = "I am excited to learn about blockchain!"
hash_result = sha256(text.encode())
#Print hash
print(hash_result.hexdigest())
