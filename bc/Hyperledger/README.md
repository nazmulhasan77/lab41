# Hyperledger Fabric Asset Transfer Project

This project demonstrates setting up a **Hyperledger Fabric test network** and running a simple **Asset Transfer API Server** using **Node.js**.

---

## Project Structure
```

HFL-Project-Asset-Transfer/
├── fabric-samples/       # Hyperledger Fabric sample network
│   └── test-network/     # Test network scripts
└── api-server/           # Express.js server for Asset Transfer
└── chaincode/            # Chaincode Written in Go

```

---


---

# Setup Instructions
## Install Prerequisites accorting to the prerequisites.txt file
## Then follow the following steps 
### 1 Clone The project and Fabric Samples
```bash
git clone https://github.com/Hasin20108/HLF-Project-Asset-Transfer.git

cd HFL-Project-Asset-Transfer/blockchain-network
export PATH=$PATH:$PWD/bin
peer version
```

### 2 Check all prerequisites 
```bash
cd ..
./check_prerequisites.sh
# if any tool missing install it before proceeding
# installing commands are in prerequisites.md file
```

### 3 Start the Test Network

```bash
cd test-network

sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 


./network.sh down
./network.sh up createChannel -ca
./network.sh deployCCAAS -ccn asset -ccp ../../chaincode -ccv 1 -ccl go

# for checking the containers
docker ps --format table{{.Names}}
```

---

## Run the API Server

```bash
# location: HLF-Project-Asset-Transfer/api-server
cd ../../api-server

npm install express @hyperledger/fabric-gateway @grpc/grpc-js

# start server
node server.js
```

---

## Shutting Down the Network

```bash
# location: HLF-Project-Asset-Transfer/fabric-samples/test-network
./network.sh down
```

---

## Summary

* Fabric binaries & Docker containers set up using `install-fabric.sh`
* Test network launched with a single channel
* Asset Transfer chaincode deployed
* Node.js API server connected to the Fabric Gateway

---

## References

* [Hyperledger Fabric Documentation](https://hyperledger-fabric.readthedocs.io/)
* [Fabric Samples GitHub](https://github.com/hyperledger/fabric-samples)

