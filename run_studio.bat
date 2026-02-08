@echo off
set "APP_DIR=%~dp0"
cd /d "%APP_DIR%"

echo Starting Qwen3-TTS Studio...
echo (Models are loading, the browser will open automatically once ready...)

:: Launch Python directly from environment
"C:\Users\educa\miniconda3\envs\qwen3-tts\python.exe" app.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Python process exited with code %ERRORLEVEL%
    pause
)

pause
