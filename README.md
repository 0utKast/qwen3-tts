# Qwen3-TTS Studio 🚀 v1.1.0

Una interfaz web moderna y potente para el nuevo ecosistema **Qwen3-TTS** de Alibaba. Optimizada tanto para GPUs NVIDIA (RTX 3090/4090) como para **Apple Silicon (Mac Mini M4)**.

Este estudio permite tres funciones principales en una sola aplicación:
1. **Premium Presets**: Voces predefinidas de alta calidad (Vivian, Ryan, Sohee, Aiden).
2. **Zero-Shot Cloning**: Clona cualquier voz a partir de un fragmento de audio de 5-10 segundos sin necesidad de transcripción.
3. **Voice Design**: Diseña voces únicas a partir de descripciones en lenguaje natural.

## 🛠️ Características Principales

- **Multi-Model Engine**: Soporte para motores estándar (Torch) y **Optimizado (MLX + uv)**.
- **Apple Silicon Native**: Integración con `mlx-audio` para latencia ultra-baja en chips M4/M3/M2.
- **uv Integration**: Uso de `uv` para una ejecución y gestión de dependencias instantánea.
- **Voice Library**: Guarda tus diseños y clones favoritos con nombres personalizados.
- **Drag-and-Drop**: Soporte para arrastrar archivos de audio y PDFs.
- **Optimización VRAM**: Gestión inteligente de memoria unificada en Mac y VRAM dedicada en NVIDIA.

## 📋 Requisitos

- **OS**: macOS (Apple Silicon), Windows (NVIDIA) o Linux.
- **Hardware**: 
  - **Mac**: Chip M-series (M4 Pro recomendado) para el motor optimizado.
  - **NVIDIA**: Al menos 12GB de VRAM (24GB recomendado).
- **Herramientas**: `uv` (recomendado para Mac), Python 3.11+

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
