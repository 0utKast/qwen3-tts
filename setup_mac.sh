#!/bin/bash

# Este script configura el entorno para ejecutar la aplicación TTS.
# Si falla, es posible que necesites instalar las Command Line Tools de Xcode manualmente:
#   xcode-select --install

echo "=== Configuración de Qwen3-TTS para Mac (Apple Silicon) ==="

# 1. Verificar Homebrew
if ! command -v brew &> /dev/null; then
    echo "❌ Homebrew no encontrado."
    echo "Por favor, instala Homebrew primero ejecutando este comando en una terminal:"
    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo "Luego vuelve a ejecutar este script."
    exit 1
fi
echo "✅ Homebrew detectado."

# 2. Instalar dependencias de sistema con Homebrew
echo "📦 Verificando/Instalando dependencias de sistema (ffmpeg, portaudio)..."
# Usamos || true para que el script continúe si ya están instalados o pide password y falla (para dar mensaje)
brew install ffmpeg portaudio python@3.11 || echo "⚠️  Advertencia: Hubo un problema instalando paquetes con brew. Si ya los tienes, ignora esto."

# 3. Configurar entorno virtual Python
echo "🐍 Configurando entorno virtual Python..."
if [ ! -d "venv" ]; then
    python3.11 -m venv venv || python3 -m venv venv
    echo "   Entorno virtual creado."
else
    echo "   Entorno virtual encontrado."
fi

# Activar entorno
source venv/bin/activate

# 4. Actualizar pip e instalar dependencias Python
echo "⬇️  Instalando librerías Python..."
pip install --upgrade pip
# Instalar PyTorch Stable (la detección automática suele funcionar en Mac)
pip install torch torchvision torchaudio
# Instalar resto de requirements
pip install -r requirements.txt

echo "✅ Configuración completada."
echo "Para verificar soporte GPU (MPS), ejecuta: python check_mps.py"
echo "Para iniciar la aplicación, ejecuta: ./run_mac.sh"
