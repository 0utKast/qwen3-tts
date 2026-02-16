#!/bin/bash

# Project Directory (Absolute Path)
PROJECT_DIR="/Users/jesusconde/MisApps/qwen3-tts-main"
cd "$PROJECT_DIR"

echo "==============================================="
echo "   Qwen3-TTS Launcher for Mac Mini Pro"
echo "==============================================="

# Environment setup
export PATH="/opt/homebrew/bin:$PATH"
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export HF_HUB_ENABLE_HF_TRANSFER=1

# Activate virtual environment
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found. Please run setup first."
    exit 1
fi

# Open the browser in the background
sleep 2 && open "http://127.0.0.1:5051" &

# Run the app
python app.py
