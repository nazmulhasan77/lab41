"""
Write a program in Python to implement PoW algorithm. 
"""


import hashlib

def mine_block(data, difficulty):
    nonce = 0
    prefix = "0"*difficulty

    while True:
        text = data + str(nonce)
        hash_val = hashlib.sha256(text.encode()).hexdigest()

        if hash_val.startswith(prefix):
            return nonce, hash_val

        nonce += 1
        print("Nonce:", nonce)


data = "Block Data"
difficulty = 4

nonce, hash_val = mine_block(data, difficulty)

print("Data:", data)
print("Nonce:", nonce)
print("Hash:", hash_val)
