export PATH="/opt/homebrew/bin:$PATH"
# MPS Tuning for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.7 # 70% of 24GB is ~17GB, leaving room for OS
export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== Iniciando Qwen3-TTS (Mac Mini Pro Optimized) ==="
source venv/bin/activate
mkdir -p static/audio/sessions
python app.py
