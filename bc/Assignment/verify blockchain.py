# verify_blockchain.py

import hashlib
import json
import os

class Blockheader:
    def __init__(self, version, difficulty, previous_hash, timestamp, merkle_root):
        self.version = version
        self.difficulty = difficulty
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.merkle_root = merkle_root

class Block:
    def __init__(self, data, blockheader, nonce, stored_hash):
        self.data = data
        self.blockheader = blockheader
        self.nonce = nonce
        self.hash = stored_hash

    def calculate_hash(self):
        hash_string = (
            str(self.data) +
            str(self.blockheader.previous_hash) +
            str(self.nonce) +
            str(self.blockheader.version) +
            str(self.blockheader.timestamp) +
            str(self.blockheader.merkle_root)
        )
        return hashlib.sha256(hash_string.encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = []

    def load_from_file(self, filename='blockchain.json'):
        if not os.path.exists(filename):
            print(f"File {filename} does not exist.")
            return False

        with open(filename, 'r') as f:
            chain_data = json.load(f)

        self.chain = []
        for block_data in chain_data:
            bh = block_data["blockheader"]
            blockheader = Blockheader(
                version=bh["version"],
                difficulty=bh["difficulty"],
                previous_hash=bh["previous_hash"],
                timestamp=bh["timestamp"],
                merkle_root=bh["merkle_root"]
            )
            block = Block(
                data=block_data["data"],
                blockheader=blockheader,
                nonce=block_data["nonce"],
                stored_hash=block_data["hash"]
            )
            self.chain.append(block)
        print(f"Blockchain loaded from {filename} ✅")
        return True

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]

            if current.blockheader.previous_hash != previous.hash:
                print(f"⛔ Invalid link at block {i}")
                return False

            recomputed_hash = current.calculate_hash()

            if current.hash != recomputed_hash:
                print(f"⛔ Invalid hash at block {i}")
                return False

            if not current.hash.startswith('0' * current.blockheader.difficulty):
                print(f"⛔ Invalid proof of work at block {i}")
                return False

        print("✅ Blockchain is authentic and untampered.")
        return True


# === MAIN USAGE ===

if __name__ == "__main__":
    chain = Blockchain()
    if chain.load_from_file('blockchain.json'):
        chain.is_valid()
