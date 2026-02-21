# 🌐 Blockchain 
## 💡 Introduction

Blockchain is a **distributed, immutable ledger** that allows participants to securely record transactions without a central authority. It’s the foundation of cryptocurrencies, supply chain management, voting systems, and decentralized applications (DApps).

In this project, you’ll learn how to:

* Create blocks and link them securely
* Validate transactions
* Implement a proof-of-work consensus mechanism
* Explore decentralized networks
* Execute simple smart contracts

---

## 📚 Blockchain Theory

### 🔹 Block Structure

A block contains:

* **Index:** Position of the block in the chain
* **Timestamp:** Time of creation
* **Transactions:** List of transactions recorded in the block
* **Previous Hash:** Hash of the previous block
* **Nonce:** Number used in mining for proof-of-work
* **Hash:** Cryptographic hash of the current block

```
Block {
  index: 1,
  timestamp: "2025-08-26T12:00:00Z",
  transactions: [...],
  previous_hash: "0000abcd...",
  nonce: 3421,
  hash: "0000ef12..."
}
```

---

### 🔹 Hashing and Cryptography

* Blockchain uses cryptographic hash functions (like **SHA-256**) to ensure immutability.
* Any change in a block changes its hash, breaking the chain.
* Hash = `SHA256(index + timestamp + transactions + previous_hash + nonce)`

---

### 🔹 Consensus Mechanisms

Blockchain relies on **consensus** to validate transactions across distributed nodes.

**Common mechanisms:**

1. **Proof of Work (PoW):** Nodes solve computational puzzles.
2. **Proof of Stake (PoS):** Validators are chosen based on stake.
3. **Delegated Proof of Stake (DPoS):** Stakeholders vote for validators.

This project implements **PoW**, which secures the network against attacks.

---

### 🔹 Mining Process

Mining is the process of adding a new block by finding a **nonce** such that the block hash meets a difficulty criterion (e.g., starts with `0000`).

**Steps:**

1. Collect transactions from the pool
2. Create a new block
3. Increment nonce until hash meets difficulty
4. Broadcast block to network

---

### 🔹 Transactions

A transaction records a transfer of value or execution of a smart contract.

**Transaction Fields:**

* Sender address
* Receiver address
* Amount/value
* Timestamp
* Signature (ensures authenticity)

---

### 🔹 Decentralization and P2P Network

* Each node in the network maintains a copy of the blockchain
* Nodes communicate via a **peer-to-peer network**
* Adding a block requires consensus from majority of nodes
* Eliminates the need for a central authority

---

### 🔹 Smart Contracts

Smart contracts are **self-executing scripts** stored on the blockchain.
They automatically execute actions when predefined conditions are met.

Example:

```python
def simple_contract(sender, receiver, amount):
    if sender.balance >= amount:
        sender.balance -= amount
        receiver.balance += amount
```

---

## ⚙️ Features

* Create and validate transactions
* Mine new blocks with Proof-of-Work
* Full chain validation
* Peer-to-peer network simulation
* Optional smart contract execution
* Detailed logs and block information

---

## 🛠 Installation

```bash
# Clone repository
git clone https://github.com/nazmulhasan77/Blockchain.git
cd Blockchain

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

```bash
# Run the blockchain node
python blockchain.py
```

---

## 📝 Examples

1. **Creating a transaction**

```python
add_transaction(sender="Alice", receiver="Bob", amount=50)
```

2. **Mining a block**

```python
mine_block()
```

3. **Viewing the blockchain**

```python
print(blockchain.chain)
```

---

## 🤝 Contributing

Contributions are welcome! Fork the repository, make improvements, and submit a pull request. Please follow the **code of conduct** and maintain clean documentation.

---

## 📜 License

This project is licensed under the **MIT License**.
