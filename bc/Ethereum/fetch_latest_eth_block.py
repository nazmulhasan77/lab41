import requests
import json
import time
import os
from datetime import datetime

# ================= CONFIGURATION =================
API_KEY = os.getenv("ETHERSCAN_API_KEY", "UV6YPXH57WEC6QQ8D9HBNQNBE6HY8VC1IB")
BASE_URL = "https://api.etherscan.io/v2/api"
CHAIN_ID = 1  # Ethereum Mainnet

# ================================================

def get_latest_block_number():
    params = {
        'chainid': CHAIN_ID,
        'module': 'proxy',
        'action': 'eth_blockNumber',
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    print(f"Raw API Response (Block Number): {json.dumps(data, indent=2)}")

    # V2 returns JSON-RPC: {"jsonrpc": "2.0", "id": ..., "result": "0x..."}
    if 'result' in data and data['result'].startswith('0x'):
        block_number = int(data['result'], 16)
        return block_number
    else:
        error = data.get('error', 'Unknown error')
        raise Exception(f"API Error: {error}")

def get_block_by_number(block_number):
    time.sleep(1)  # Respect rate limits
    params = {
        'chainid': CHAIN_ID,
        'module': 'proxy',
        'action': 'eth_getBlockByNumber',
        'tag': hex(block_number),
        'boolean': 'true',
        'apikey': API_KEY
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()

    print(f"Raw API Response (Block Details): {json.dumps(data, indent=2)}")

    if 'result' in data and isinstance(data['result'], dict):
        return data['result']
    else:
        error = data.get('error', 'Unknown error')
        raise Exception(f"API Error: {error}")

def format_timestamp(hex_timestamp):
    timestamp = int(hex_timestamp, 16)
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')

def main():
    try:
        print("Fetching latest block number (Etherscan V2 API)...")
        latest_block_num = get_latest_block_number()
        print(f"Latest Block Number: {latest_block_num}")

        print("\nFetching block details...")
        block = get_block_by_number(latest_block_num)

        # Format key info
        block_info = {
            "Block Number": int(block['number'], 16),
            "Hash": block['hash'],
            "Parent Hash": block['parentHash'],
            "Miner": block['miner'],
            "Timestamp": format_timestamp(block['timestamp']),
            "Gas Used": int(block['gasUsed'], 16),
            "Gas Limit": int(block['gasLimit'], 16),
            "Transaction Count": len(block['transactions']),
            "Base Fee Per Gas": int(block.get('baseFeePerGas', '0x0'), 16) if 'baseFeePerGas' in block else "N/A",
        }

        print("\n" + "="*60)
        print("LATEST ETHEREUM BLOCK (via Etherscan V2)")
        print("="*60)
        for key, value in block_info.items():
            print(f"{key:<20}: {value}")

        if block['transactions']:
            print(f"\nFirst 2 Transaction Hashes:")
            for tx in block['transactions'][:2]:
                print(f"  - {tx['hash']}")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()