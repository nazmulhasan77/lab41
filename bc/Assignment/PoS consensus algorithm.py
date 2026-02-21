"""
Write a program in Python to implement PoS consensus algorithm.
"""


import random

validators = {
    "Alice": 50,
    "Bob": 30,
    "Charlie": 20
}

def select_validator():
    total = sum(validators.values())
    pick = random.uniform(0,total)
    current = 0

    for v,stake in validators.items():
        current += stake
        if current > pick:
            return v

selected = select_validator()
print("Selected Validator:", selected)
