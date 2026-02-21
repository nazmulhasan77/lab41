"""
Write a program in Python to implement blockchain.
"""


"""
step 1: block creation
    hash = index+timestamp+data+previous hash

step 2: blockchain creation (class or list, here we use class)
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
        value = str(self.index) + str(self.timestamp) + str(self.data) + str(self.previous_hash)
        return hashlib.sha256(value.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, "Genesis Block", "0")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        prev_block = self.get_latest_block()
        new_block = Block(len(self.chain), data, prev_block.hash)
        self.chain.append(new_block)


# -------- Test --------
bc = Blockchain()
bc.add_block("First block data")
bc.add_block("Second block data")

for block in bc.chain:
    print("Index:", block.index)
    print("Timestamp:", block.timestamp)
    print("Data:", block.data)
    print("Prev Hash:", block.previous_hash)
    print("Hash:", block.hash)
    
    print("-"*40)   # optional : just print separator for better readability
