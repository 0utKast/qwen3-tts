# Qwen3-TTS Studio 🚀

Una interfaz web moderna y potente para el nuevo ecosistema **Qwen3-TTS** de Alibaba. Diseñada específicamente para aprovechar la potencia de las GPUs NVIDIA (especialmente optimizada para RTX 3090/4090 con 24GB de VRAM).

Este estudio permite tres funciones principales en una sola aplicación:
1. **Premium Presets**: Voces predefinidas de alta calidad (Vivian, Ryan, Sohee, Aiden).
2. **Zero-Shot Cloning**: Clona cualquier voz a partir de un fragmento de audio de 5-10 segundos sin necesidad de transcripción.
3. **Voice Design**: Diseña voces únicas a partir de descripciones en lenguaje natural.

## 🛠️ Características Principales

- **Multi-Model Engine**: Carga simultáneamente las variantes `Base`, `CustomVoice` y `VoiceDesign` en la GPU para cambios instantáneos.
- **Voice Library**: Guarda tus diseños y clones favoritos con nombres y descripciones personalizadas.
- **Drag-and-Drop**: Soporte para arrastrar archivos de audio para clonación rápida.
- **Procesamiento PDF**: Extrae texto de archivos PDF para lectura masiva.
- **Optimización RTX 3090**: Gestión de memoria optimizada para 24GB VRAM, evitando fragmentación y maximizando la velocidad.

## 📋 Requisitos

- **OS**: Windows (probado en Windows 11) o Linux (WSL2 recomendado).
- **GPU**: NVIDIA con al menos 12GB de VRAM (24GB recomendado para carga triple de modelos).
- **Python**: 3.8+
- **Conda**: Recomendado para la gestión del entorno.

## 🚀 Instalación y Configuración

Sigue estos pasos para instalar el estudio en tu equipo:

### 1. Clonar el repositorio
```bash
git clone https://github.com/0utKast/qwen3-tts.git
cd qwen3-tts
```

### 2. Crear el entorno virtual (Conda)
```bash
conda create -n qwen3-tts python=3.10 -y
conda activate qwen3-tts
```

### 3. Instalar dependencias
Primero, instala el core de Qwen3-TTS (puedes encontrarlo en el repo oficial de Alibaba) y luego:
```bash
pip install -r requirements.txt
```

> [!IMPORTANT]
> Asegúrate de tener instalada la versión de PyTorch compatible con tu versión de CUDA.
> Recomendado: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

### 4. Lanzar la aplicación
Puedes usar el script incluido:
```bash
run_studio.bat
```

## 📖 Guía de Uso

### Clonación de Voz (Zero-Shot)
1. Ve a la pestaña **Voice Clone**.
2. Arrastra un archivo `.wav` o `.mp3` (recomendado 5-10 segundos, audio limpio).
3. Escribe el texto y pulsa "Generate".
4. Si te gusta el resultado, pulsa **"Save this Clone"** para añadirlo a tu biblioteca permanente.

### Diseño de Voz
1. En la pestaña **Voice Design**, describe la voz que quieres (ej: "A mature male voice with a deep, calm tone, slightly raspy").
2. Genera y guarda si el resultado es satisfactorio.

### Biblioteca de Voces
Tus voces guardadas se almacenan en `voices.json` y los audios de referencia en `static/audio/permanent_voices/`. Puedes borrarlos manualmente o editarlos en el archivo JSON.

## 🛡️ Licencia
Este proyecto utiliza los modelos de Alibaba Qwen. Consulta la licencia original de Qwen3-TTS para más detalles sobre el uso comercial.

---
Creado con ❤️ para la comunidad de IA.
