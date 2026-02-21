"""
Write a program in Python to implement a blockchain and UTXo (unspent transaction output).
"""

class Transaction:
    def __init__(self, sender, receiver, amount):
        self.sender = sender
        self.receiver = receiver
        self.amount = amount

class BlockchainUTXO:
    def __init__(self):
        self.utxo = {}   # wallet -> balance
        self.chain = []

    def create_transaction(self, tx):
        if tx.sender != "SYSTEM":
            if self.utxo.get(tx.sender,0) < tx.amount:
                print("Transaction failed: insufficient balance")
                return

            self.utxo[tx.sender] -= tx.amount

        self.utxo[tx.receiver] = self.utxo.get(tx.receiver,0) + tx.amount
        self.chain.append(tx)
        print("Transaction successful")

    def show_balances(self):
        print("\nBalances:")
        for user,balance in self.utxo.items():
            print(user,":",balance)


bc = BlockchainUTXO()

bc.create_transaction(Transaction("SYSTEM","Alice",50))
bc.create_transaction(Transaction("Alice","Bob",20))
bc.create_transaction(Transaction("Bob","Charlie",10))

bc.show_balances()
