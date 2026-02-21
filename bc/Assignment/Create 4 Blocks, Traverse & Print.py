"""
Write a program in Python to create four new blocks in a blockchain. Traverse the blocks and
print the values
"""

"""
step 1: block creation
    hash = index+timestamp+data+previous hash

step 2: blockchain creation (class or list, here we use list)
    create genesis block (first block with index 0 and previous hash 0)
    add new blocks with reference to previous block's hash
    
step 3: traverse and print blocks
"""

import hashlib
import time

class Block:
    def __init__(self, index, data, previous_hash):
        self.index = index
        self.timestamp = time.time()
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        value = str(self.index)+str(self.timestamp)+str(self.data)+str(self.previous_hash)
        return hashlib.sha256(value.encode()).hexdigest()


# -------- Create Blocks --------
blockchain = []

genesis = Block(0, "Genesis Block", "0")
blockchain.append(genesis)

for i in range(1,5):
    prev = blockchain[-1]
    block = Block(i, f"Block {i} Data", prev.hash)
    blockchain.append(block)


# -------- Traverse --------
print("Traversing Blockchain:\n")

for block in blockchain:
    print("Index:", block.index)
    print("Timestamp:", block.timestamp)
    print("Data:", block.data)
    print("Prev Hash:", block.previous_hash)
    print("Hash:", block.hash)
    
    print("-"*40)
