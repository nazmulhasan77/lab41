"""
Write a program in Python that Demonstrates How to Use the SHA-256 Hash Function and Its
Application in a Simple Blockchain
"""


import hashlib
import time

class Block:
    def __init__(self, data, prev_hash):
        self.timestamp = time.time()
        self.data = data
        self.prev_hash = prev_hash
        self.hash = self.generate_hash()

    def generate_hash(self):
        text = str(self.timestamp) + self.data + self.prev_hash
        return hashlib.sha256(text.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [Block("Genesis Block","0")]

    def add_block(self,data):
        prev = self.chain[-1]
        new_block = Block(data, prev.hash)
        self.chain.append(new_block)

    def show(self):
        for i,block in enumerate(self.chain):
            print("\nBlock",i)
            print("Data:", block.data)
            print("Timestamp:", block.timestamp)
            print("Prev Hash:", block.prev_hash)
            print("Hash:", block.hash)


bc = Blockchain()
bc.add_block("Alice pays Bob 10")
bc.add_block("Bob pays Charlie 5")

bc.show()
