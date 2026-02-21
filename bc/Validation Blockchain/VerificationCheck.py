import hashlib
import json
import time


class Blockheader:
    def __init__(self, version, difficulty, previous_hash="0", merkle_root="dummy_merkle_root", timestamp=None):
        self.version = version
        self.difficulty = difficulty
        self.previous_hash = previous_hash
        self.merkle_root = merkle_root
        self.timestamp = timestamp if timestamp else int(time.time())


class Block:
    def __init__(self, data, blockheader):
        self.data = data
        self.blockheader = blockheader
        self.nonce = 0
        self.hash = ""
        self.mine(self.blockheader.difficulty)

    def compute_hash(self):
        """Recompute hash of block from its contents."""
        hash_string = (
            str(self.data) +
            str(self.blockheader.previous_hash) +
            str(self.blockheader.merkle_root) +
            str(self.blockheader.timestamp) +
            str(self.nonce) +
            str(self.blockheader.version)
        )
        return hashlib.sha256(hash_string.encode()).hexdigest()

    def mine(self, difficulty):
        """Perform Proof-of-Work mining until hash matches difficulty."""
        while self.hash[0:difficulty] != '0' * difficulty:
            self.nonce += 1
            self.hash = self.compute_hash()
            print(f"Mining... Nonce: {self.nonce}, Hash: {self.hash}", end="\r")
        print(f"\nBlock mined successfully: {self.hash}")
        return self.hash

    def serialize(self):
        block_dict = {
            "data": self.data,
            "blockheader": self.blockheader.__dict__,
            "nonce": self.nonce,
            "hash": self.hash
        }
        return block_dict  


class Blockchain:
    def __init__(self, difficulty=4):
        self.chain = []
        self.difficulty = difficulty
        self.create_genesis_block()

    def create_genesis_block(self):
        """First block in the blockchain."""
        genesis_header = Blockheader(version=1, difficulty=self.difficulty, previous_hash="0")
        genesis_block = Block("Genesis Block", genesis_header)
        self.chain.append(genesis_block)

    def add_block(self, data):
        """Add new block with given transaction data."""
        previous_hash = self.chain[-1].hash
        blockheader = Blockheader(version=1, difficulty=self.difficulty, previous_hash=previous_hash)
        new_block = Block(data, blockheader)
        self.chain.append(new_block)

    def serialize_chain(self):
        """Return blockchain as JSON string."""
        blockchain_dict = [block.serialize() for block in self.chain]
        return json.dumps(blockchain_dict, indent=1)

    def is_chain_valid(self):
        """Validate blockchain integrity."""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            prev_block = self.chain[i - 1]

            # Recompute hash of current block
            recalculated_hash = current_block.compute_hash()

            # Check if stored hash matches recalculated hash
            if current_block.hash != recalculated_hash:
                print(f"❌ Tampering detected at Block {i}: hash mismatch!")
                return False

            # Check if previous_hash matches actual previous block's hash
            if current_block.blockheader.previous_hash != prev_block.hash:
                print(f"❌ Tampering detected at Block {i}: previous hash mismatch!")
                return False

        print("✅ Blockchain integrity verified. No tampering detected.")
        return True


# ---------------- Example usage ---------------- #
if __name__ == "__main__":
    my_blockchain = Blockchain(difficulty=3)
    my_blockchain.add_block("First real block")
    my_blockchain.add_block("Second real block")
    my_blockchain.add_block("Third real block")
    my_blockchain.add_block("Fourth real block")
    my_blockchain.add_block("Fifth real block")

    print("\n=== Blockchain Data ===")
    print(my_blockchain.serialize_chain())

    print("\n=== Authenticating Blockchain ===")
    my_blockchain.is_chain_valid()

    # 🔴 Simulate tampering
    print("\n--- Tampering with Block 3 ---")
    my_blockchain.chain[2].data = "Hacked Transaction!"
    my_blockchain.is_chain_valid()
