"""
Write a program in Python to Fetch the Latest Block Information from Ethereum Blockchain Using
Etherscan API
"""


import requests

API_KEY = "YOUR_API_KEY_HERE"

url = f"https://api.etherscan.io/api?module=proxy&action=eth_blockNumber&apikey={API_KEY}"

res = requests.get(url)
block_hex = res.json()["result"]

block_number = int(block_hex,16)

print("Latest Block Number:", block_number)


"""
pre requisite coomand: pip install requests

"""