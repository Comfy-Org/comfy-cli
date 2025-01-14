#!/bin/bash

# Flag to track if yum update has been run
update_flag=false

# Check if ~/comfy directory exists, create it if not
if [ ! -d "$(pwd)/comfy" ]; then
    mkdir "$(pwd)/comfy"
fi

echo "[1/6] CHECK: python3"

# Check if Python3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Starting installation..."
    if [ "$(id -u)" -eq 0 ]; then # Check if running as root
        yum update -y
        yum install -y python3
    else
        sudo yum update -y
        sudo yum install -y python3
    fi
    update_flag=true
fi

echo "[2/6] CHECK: git"

# Check if Git is installed
if ! command -v git &> /dev/null; then
    echo "Git is not installed. Starting installation..."
    if [ "$update_flag" == false ]; then
        if [ "$(id -u)" -eq 0 ]; then # Check if running as root
            yum update -y
        else
            sudo yum update -y
        fi
        update_flag=true
    fi
    if [ "$(id -u)" -eq 0 ]; then # Check if running as root
        yum install -y git-core
    else
        sudo yum install -y git-core
    fi
fi

echo "[3/6] CREATE: venv (~/comfy/venv)"

# Create virtual environment
if [ ! -d "$(pwd)/comfy/venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$(pwd)/comfy/venv"
fi

echo "[4/6] INSTALL: comfy-cli into venv"

# Install Comfy-CLI
echo "Installing Comfy-CLI..."
source "$(pwd)/comfy/venv/bin/activate"
pip install comfy-cli
comfy --install-completion

echo "[5/6] INSTALL: ComfyUI into $(pwd)/comfy/ComfyUI"

# Run comfy install
echo "Running comfy install..."
comfy install

echo "[6/6] CREATE: Script for 'comfy launch'"
if [ ! -f "$(pwd)/comfy/run.sh" ]; then
    echo "source $(pwd)/comfy/venv/bin/activate" > "$(pwd)/comfy/run.sh"
    echo "comfy launch -- \$*" >> "$(pwd)/comfy/run.sh"
    chmod +x "$(pwd)/comfy/run.sh"
else
    echo "Script file already exists: ~/comfy/run.sh"
fi

# Print virtual environment path
echo "==========================================================="
echo "Virtual environment path: $(pwd)/comfy/venv"
echo "ComfyUI path: $(pwd)/comfy/ComfyUI"
echo "Default launch script path: $(pwd)/comfy/run.sh"
echo "DONE."