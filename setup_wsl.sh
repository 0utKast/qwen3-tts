#!/bin/bash
# setup_wsl.sh
# Automated setup for Qwen3-TTS on WSL Ubuntu

echo "Starting WSL Environment Setup for Qwen3-TTS..."

# 1. Update system and check for basic tools
echo "Checking system dependencies..."
# We need build-essential and cuda-toolkit for compiling flash-attn
if ! dpkg -l | grep -q "cuda-toolkit"; then
    echo "Installing CUDA Toolkit and build-essential..."
    sudo apt update
    sudo apt install -y build-essential nvidia-cuda-toolkit
fi

which wget > /dev/null || (echo "Error: wget not found. Please run 'sudo apt update && sudo apt install wget' and try again." && exit 1)


# 2. Install Miniconda if not present
if ! command -v conda &> /dev/null
then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
    conda init bash
    echo "Miniconda installed. Please RESTART your Ubuntu terminal and run this script again to continue."
    exit 0
fi

# 2.5 Accept Anaconda Terms of Service (New requirement in recent versions)
echo "Accepting Anaconda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# 3. Create and activate environment
ENV_NAME="qwen3-tts"
if ! conda info --envs | grep -q "$ENV_NAME"
then
    echo "Creating conda environment: $ENV_NAME..."
    conda create -y -n $ENV_NAME python=3.10 || (echo "Error: Failed to create conda environment." && exit 1)
fi


# Use the full path to the env python to ensure we are in the right place
ENV_PYTHON="$HOME/miniconda/envs/$ENV_NAME/bin/python"
ENV_PIP="$HOME/miniconda/envs/$ENV_NAME/bin/pip"

echo "Installing PyTorch with CUDA support..."
$ENV_PIP install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Setup CUDA environment variables for compilation
# Detect CUDA_HOME dynamically
if command -v nvcc &> /dev/null; then
    # If nvcc is in /usr/bin/nvcc, dirname(dirname) is /usr
    # and /usr/bin/nvcc exists, which is what flash-attn wants.
    REAL_NVCC=$(readlink -f $(which nvcc))
    export CUDA_HOME=$(dirname $(dirname $REAL_NVCC))
    echo "Detected CUDA_HOME: $CUDA_HOME"
else
    export CUDA_HOME=/usr/lib/cuda
fi
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH


echo "Installing project requirements..."
$ENV_PIP install -r requirements.txt
$ENV_PIP install ninja  # Recommended for flash-attn build

echo "Installing Flash Attention (this may take a few minutes)..."
# We use no-build-isolation to use the already installed torch/ninja
$ENV_PIP install flash-attn --no-build-isolation


echo "Setup Complete!"
echo "To run the app, use: bash run_wsl.sh"
