#!/usr/bin/env bash

# ============================================================
#  Prerequisite Check Script
#  This script checks whether required tools are installed.
#  It DOES NOT install or uninstall anything.
# ============================================================

set -e

BLUE="\e[34m"
GREEN="\e[32m"
RED="\e[31m"
YELLOW="\e[33m"
NC="\e[0m"

divider() {
    echo -e "${BLUE}------------------------------------------------------------${NC}"
}

check_cmd() {
    if command -v "$1" >/dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} $1 is installed"
        if [[ -n "$2" ]]; then
            echo "Version: $($2)"
        fi
    else
        echo -e "${RED}[MISSING]${NC} $1 is NOT installed"
    fi
    echo
}

echo -e "${BLUE}========== Checking Prerequisites ==========${NC}"

# ------------------------------------------------------------
divider
echo -e "${YELLOW}Checking System Base Packages (ca-certificates, gnupg, lsb-release)...${NC}"

for pkg in ca-certificates gnupg lsb-release; do
    if dpkg -l | grep -q "^ii  $pkg"; then
        echo -e "${GREEN}[OK]${NC} $pkg installed"
    else
        echo -e "${RED}[MISSING]${NC} $pkg not installed"
    fi
done
echo

# ------------------------------------------------------------
divider
echo -e "${YELLOW}Checking curl...${NC}"
check_cmd curl "curl --version | head -n1"

# ------------------------------------------------------------
divider
echo -e "${YELLOW}Checking Docker & Docker Desktop...${NC}"

if command -v docker >/dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} docker installed"
    echo "Docker version: $(docker --version)"
    echo

    # Test docker permissions
    if groups $USER | grep -q docker; then
        echo -e "${GREEN}[OK]${NC} user has docker group permissions"
    else
        echo -e "${RED}[MISSING]${NC} user NOT in docker group"
    fi

    # Test docker run
    if docker info >/dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} docker daemon is running"
    else
        echo -e "${RED}[ERROR]${NC} docker daemon NOT running"
    fi
else
    echo -e "${RED}[MISSING]${NC} docker NOT installed"
fi
echo

# ------------------------------------------------------------
divider
echo -e "${YELLOW}Checking jq...${NC}"
check_cmd jq "jq --version"

# ------------------------------------------------------------
divider
echo -e "${YELLOW}Checking Go...${NC}"
check_cmd go "go version"

# ------------------------------------------------------------
divider
echo -e "${YELLOW}Checking Node.js & npm...${NC}"

if command -v node >/dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} node installed"
    node --version
else
    echo -e "${RED}[MISSING]${NC} node NOT installed"
fi

if command -v npm >/dev/null 2>&1; then
    echo -e "${GREEN}[OK]${NC} npm installed"
    npm --version
else
    echo -e "${RED}[MISSING]${NC} npm NOT installed"
fi
echo

# ------------------------------------------------------------
divider
echo -e "${GREEN}Prerequisite check complete.${NC}"
divider