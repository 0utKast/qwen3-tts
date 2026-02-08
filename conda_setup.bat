@echo off
set "CONDA_PATH=C:\Users\educa\miniconda3\Scripts\conda.exe"
set "ENV_NAME=qwen3-tts"

echo [1/3] Creating Conda environment '%ENV_NAME%' with Python 3.11...
"%CONDA_PATH%" create -n %ENV_NAME% python=3.11 -y

echo [2/3] Installing PyTorch with CUDA 12.1 support...
call C:\Users\educa\miniconda3\condabin\activate.bat %ENV_NAME%
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall

echo [3/3] Installing other requirements...
pip install -r requirements.txt

echo.
echo Setup complete! You can now use the 'Qwen3-TTS Studio' shortcut.
pause
