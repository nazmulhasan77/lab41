"""
Write a Program in Python to Verify Hash Properties
"""

import hashlib

msg1 = input("Enter first message: ")
msg2 = input("Enter second message: ")

hash1 = hashlib.sha256(msg1.encode()).hexdigest()
hash2 = hashlib.sha256(msg2.encode()).hexdigest()

print("\nHash 1:", hash1)
print("Hash 2:", hash2)

print("\nLength of Hash:", len(hash1))

if hash1 == hash2:
    print("Hashes are equal (rare collision)")
else:
    print("Hashes are different → Small change = Big difference")
