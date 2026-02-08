#!/bin/bash
# run_wsl.sh
# Script to run Qwen3-TTS in WSL Ubuntu

ENV_NAME="qwen3-tts"
PYTHON_CMD="$HOME/miniconda/envs/$ENV_NAME/bin/python"

if [ ! -f "$PYTHON_CMD" ]; then
    echo "Error: Environment $ENV_NAME not found. Please run 'bash setup_wsl.sh' first."
    exit 1
fi

echo "Launching Qwen3-TTS in WSL..."
# Allow connections from Windows (host)
$PYTHON_CMD app.py
