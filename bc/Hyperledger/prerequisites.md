## Always start clean
``` bash
sudo apt update
```
## Install curl
``` bash
sudo apt install -y curl
curl --version
```



## Docker Installation

### install docker desktop from the following link
```bash
https://docs.docker.com/desktop/setup/install/linux/ubuntu/
```

### grant permission for docker
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker 
docker ps 
docker run hello-world
```


## Install jq
```bash
sudo apt install -y jq
jq --version
```

## Install Go
```bash
sudo apt install -y golang-go
go version
```

## Install node.js
``` bash
curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash -
sudo apt install -y nodejs
```



