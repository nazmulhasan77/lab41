"""
Write a program in Python to implement a blockchain and print the values of all fields as
described in etherscan.io
"""

import hashlib
import time

class Block:
    def __init__(self, number, transactions, previous_hash):
        self.blockNumber = number
        self.timestamp = time.ctime()
        self.transactions = transactions
        self.previousHash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        value = str(self.blockNumber)+self.timestamp+str(self.transactions)+self.previousHash+str(self.nonce)
        return hashlib.sha256(value.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, ["Genesis Block"], "0")

    def add_block(self, transactions):
        prev = self.chain[-1]
        block = Block(len(self.chain), transactions, prev.hash)
        self.chain.append(block)

    def display(self):
        for block in self.chain:
            print("\nBlock Number:", block.blockNumber)
            print("Timestamp:", block.timestamp)
            print("Transactions:", block.transactions)
            print("Previous Hash:", block.previousHash)
            print("Nonce:", block.nonce)
            print("Hash:", block.hash)


bc = Blockchain()
bc.add_block(["A->B 5 BTC", "C->D 2 BTC"])
bc.add_block(["E->F 1 BTC"])

bc.display()
