"""
Write a Python program to Demonstrate the Mining Process in Blockchain 
"""


import hashlib

data = "Sample Block"
difficulty = 4

nonce = 0
prefix = "0"*difficulty

while True:
    text = data + str(nonce)
    hash_val = hashlib.sha256(text.encode()).hexdigest()

    if hash_val.startswith(prefix):
        break

    nonce += 1

print("Block Data:", data)
print("Nonce Found:", nonce)
print("Hash:", hash_val)
