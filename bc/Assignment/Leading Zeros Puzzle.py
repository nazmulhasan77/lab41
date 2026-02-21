"""
Write a Python Program that Takes a String and the Desired Number of Leading Zeros from the
User and Outputs the Input String, the Nonce Value for Which the Leading Zeros Puzzle Is Solved,
and the Corresponding Hash Generated
"""


import hashlib

text = input("Enter string: ")
difficulty = int(input("Enter number of leading zeros: "))

nonce = 0
prefix = "0" * difficulty

while True:
    data = text + str(nonce)
    hash_val = hashlib.sha256(data.encode()).hexdigest()

    if hash_val.startswith(prefix):
        break

    nonce += 1

print("\nSolution Found!")
print("Input String:", text)
print("Nonce:", nonce)
print("Hash:", hash_val)
