#!/bin/bash
# Flag to track if apt-get update has been run
update_flag=false

# Check if ~/comfy directory exists, create it if not
if [ ! -d $(pwd)/comfy ]; then
    mkdir $(pwd)/comfy
fi

echo "[1/7] CHECK: python3"
# Check if Python3 is installed
if ! command -v python3 &> /dev/null
then
    echo "Python3 is not installed. Starting installation..."
    if [ $(id -u) -eq 0 ]  # Check if running as root
    then
        apt-get update
        apt-get install -y python3
    else
        sudo apt-get update
        sudo apt-get install -y python3
    fi

    update_flag=true
fi

echo "[2/7] CHECK: python3-venv"
# Check if Python venv is installed
if ! dpkg -l | grep -qE 'python3\.[0-9]+-venv'; then
    echo "Python3-venv is not installed. Starting installation..."
    if [ "$update_flag" == false ]
    then
        if [ $(id -u) -eq 0 ]  # Check if running as root
        then
            apt-get update
        else
            sudo apt-get update
        fi

        update_flag=true
    fi
    if [ $(id -u) -eq 0 ]  # Check if running as root
    then
        apt-get install -y python3-venv
    else
        sudo apt-get install -y python3-venv
    fi
fi

echo "[3/7] CHECK: git"
# Check if Git is installed
if ! command -v git &> /dev/null
then
    echo "Git is not installed. Starting installation..."
    if [ "$update_flag" == false ]
    then
        if [ $(id -u) -eq 0 ]  # Check if running as root
        then
            apt-get update
        else
            sudo apt-get update
        fi

        update_flag=true
    fi

    if [ $(id -u) -eq 0 ]  # Check if running as root
    then
        apt-get install -y git
    else
        sudo apt-get install -y git
    fi
fi

echo "[4/7] CREATE: venv (~/comfy/venv)"
# Create virtual environment
if [ ! -d $(pwd)/comfy/venv ]
then
    echo "Creating Python virtual environment..."
    python3 -m venv $(pwd)/comfy/venv
fi

echo "[5/7] INSTALL: comfy-cli into venv"
# Install Comfy-CLI
echo "Installing Comfy-CLI..."
source $(pwd)/comfy/venv/bin/activate
pip install comfy-cli
comfy --install-completion

echo "[6/7] INSTALL: ComfyUI into $(pwd)/comfy/ComfyUI"
# Run comfy install
echo "Running comfy install..."
comfy install

echo "[7/7] CREATE: Script for 'comfy launch'"
if [ ! -f $(pwd)/comfy/run.sh ]
then
    echo "source $(pwd)/comfy/venv/bin/activate" > $(pwd)/comfy/run.sh
    echo "comfy launch -- \$*" >> $(pwd)/comfy/run.sh
    chmod +x $(pwd)/comfy/run.sh
else
    echo "Script file already exists: ~/comfy/run.sh"
fi

# Print virtual environment path
echo "==========================================================="
echo "Virtual environment path: $(pwd)/comfy/venv"
echo "ComfyUI path: $(pwd)/comfy/ComfyUI"
echo "Default launch script path: $(pwd)/comfy/run.sh"
echo "DONE."
