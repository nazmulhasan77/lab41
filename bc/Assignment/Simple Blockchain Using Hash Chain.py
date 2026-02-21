"""
Write a Python program to Demonstrate a Simple Implementation of a Blockchain Using Hash
Codes as a Chain of Blocks
"""


import hashlib

class Block:
    def __init__(self, data, prev_hash):
        self.data = data
        self.prev_hash = prev_hash
        self.hash = self.create_hash()

    def create_hash(self):
        text = self.data + self.prev_hash
        return hashlib.sha256(text.encode()).hexdigest()


# Create Blockchain
genesis = Block("Genesis Block", "0")
block1 = Block("Block 1 Data", genesis.hash)
block2 = Block("Block 2 Data", block1.hash)

chain = [genesis, block1, block2]

# Display Chain
for i, block in enumerate(chain):
    print("\nBlock", i)
    print("Data:", block.data)
    print("Prev Hash:", block.prev_hash)
    print("Hash:", block.hash)
