# Qwen3-TTS Studio 🚀 (v1.3.0)

Una interfaz web moderna y potente para el nuevo ecosistema **Qwen3-TTS** de Alibaba. Optimizada para ofrecer el mejor rendimiento tanto en sistemas NVIDIA (CUDA) como en Apple Silicon (Mac M1/M2/M3/M4) mediante aceleración nativa (MPS y MLX).

## 🌟 Novedades en v1.3.0

- **Multi-Engine Support**: Soporte tanto para el motor estándar (Torch) como el **Optimizado (MLX + uv)** para chips Apple.
- **Soporte Oficial para Mac**: Optimizado para Apple Silicon mediante MPS (Metal Performance Shaders) y latencia ultra-baja.
- **Streaming en Tiempo Real**: El audio comienza a reproducirse en cuanto el primer fragmento está listo.
- **Consistencia en Diseño de Voz**: Implementado el patrón "diseñar una vez, clonar siempre" para voces estables en textos largos.
- **Fase de Warmup**: Eliminado el retardo inicial mediante pre-calentamiento de shaders al arrancar.

## 🛠️ Características Principales

1. **Premium Presets**: Voces predefinidas de alta calidad (Vivian, Ryan, Sohee, Aiden) y voces recuperadas (Suárez, Carrillo, Nuria).
2. **Zero-Shot Cloning**: Clona cualquier voz a partir de un fragmento de audio de 5-10 segundos.
3. **Voice Design**: Diseña voces únicas a partir de descripciones en lenguaje natural.
   - *Tip*: Las descripciones en inglés (ej: "Deep pirate voice") ofrecen mayor precisión.
4. **Integration with uv**: Uso opcional de `uv` para una gestión de dependencias instantánea en Mac.
5. **Biblioteca de Voces**: Guarda y organiza tus diseños favoritos.
6. **Procesamiento PDF**: Lector integrado para documentos extensos.

## 📋 Requisitos

- **OS**: Mac (Apple Silicon) o Windows/Linux (NVIDIA).
- **Hardware**:
  - **Mac**: Chip M1 o superior (M4 Pro recomendado).
  - **NVIDIA**: Al menos 12GB de VRAM (24GB para carga triple de modelos).
- **Python**: 3.10+ (o `uv` en Mac).

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone https://github.com/0utKast/qwen3-tts.git
cd qwen3-tts
```

### 2. Entorno Virtual
```bash
# Mac / Linux
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Windows
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Lanzar la aplicación
En Mac, puedes usar el lanzador incluido:
```bash
./Qwen3-TTS.command
```
O manualmente:
```bash
python3 app.py
```

## 📖 Guía de Uso del Diseño de Voz
Para obtener los mejores resultados de identidad vocal:
1. Usa descripciones claras en inglés (ej: "A gravelly old man voice, wise and calm").
2. El sistema diseñará la identidad una vez y la mantendrá consistente durante toda la generación.

## 🛡️ Licencia
Este proyecto utiliza los modelos de Alibaba Qwen. Consulta la licencia original de Qwen3-TTS para más detalles sobre el uso comercial.

---
Creado con ❤️ para la comunidad de IA.
