import os
# Qwen3-TTS Studio v1.3.0
# Optimize Hugging Face environment
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import json
import torch
import soundfile as sf
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
import numpy as np
import uuid
import threading
import time
import fitz
import re
import shutil
import webbrowser
import gc

app = Flask(__name__)
CORS(app)

@app.errorhandler(Exception)
def handle_exception(e):
    # Error as JSON instead of HTML to avoid frontend parsing errors
    error_msg = str(e)
    print(f"--- GLOBAL ERROR: {error_msg} ---")
    import traceback
    traceback.print_exc()
    return jsonify({"error": error_msg}), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found (404)"}), 404

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = os.path.join(BASE_DIR, "static/audio")
SAVED_VOICES_FILE = os.path.join(BASE_DIR, "voices.json")
CUSTOM_VOICES_DIR = os.path.join(OUTPUT_DIR, "permanent_voices")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CUSTOM_VOICES_DIR, exist_ok=True)

def load_saved_voices():
    if os.path.exists(SAVED_VOICES_FILE):
        try:
            with open(SAVED_VOICES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_voice_to_json(voice_data):
    voices = load_saved_voices()
    voices.append(voice_data)
    with open(SAVED_VOICES_FILE, 'w', encoding='utf-8') as f:
        json.dump(voices, f, ensure_ascii=False, indent=2)

# Language mapping: UI codes -> Model expected names
LANGUAGE_MAP = {
    'zh': 'chinese',
    'en': 'english',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'ja': 'japanese',
    'ko': 'korean',
    'pt': 'portuguese',
    'ru': 'russian',
    'es': 'spanish',
    'auto': 'auto'
}

# Check for Hardware Acceleration (CUDA/MPS)
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

print(f"\n" + "-"*50)
print(f"Detected Device: {device.upper()}")
if device == "cuda":
    print("Using NVIDIA GPU (CUDA).")
elif device == "mps":
    print("Using Apple Silicon GPU (MPS).")
else:
    print("Using CPU. Warning: Generation will be slow.")
print("-"*50 + "\n")

# Initialize models
tts_custom = None
tts_base = None
voice_designer = None
models_ready = False
loading_error = None

def load_models_async():
    global tts_custom, tts_base, voice_designer, models_ready, loading_error
    try:
        print(f"--- Loading model {CUSTOM_MODEL_ID}... ---")
        tts_custom = Qwen3TTSModel.from_pretrained(
            CUSTOM_MODEL_ID, 
            device_map=device, 
            dtype=torch.float16 if device == "mps" else torch.bfloat16,
            attn_implementation="sdpa" if device == "mps" else None
        )
        print(f"--- Custom model loaded to {device} ---")
        
        print(f"--- Loading model {BASE_MODEL_ID}... ---")
        tts_base = Qwen3TTSModel.from_pretrained(
            BASE_MODEL_ID, 
            device_map=device, 
            dtype=torch.float16 if device == "mps" else torch.bfloat16,
            attn_implementation="sdpa" if device == "mps" else None
        )
        print(f"--- Base model loaded to {device} ---")
        
        print(f"--- Loading model {DESIGN_MODEL_ID}... ---")
        voice_designer = Qwen3TTSModel.from_pretrained(
            DESIGN_MODEL_ID, 
            device_map=device, 
            dtype=torch.float16 if device == "mps" else torch.bfloat16,
            attn_implementation="sdpa" if device == "mps" else None
        )
        print(f"--- Voice Designer loaded to {device} ---")
        
        # Warmup Phase: Pre-compile MPS Kernels
        if device == "mps":
            print("--- Warmup Phase: Pre-compiling MPS shaders (this prevents initial hang)... ---")
            with torch.no_grad():
                # Tiny text to trigger compilation
                tts_custom.generate_custom_voice("Warmup.", speaker="vivian", language="english")
            print("--- Warmup Complete. ---")
            
        models_ready = True
        # Open browser automatically once ready
        webbrowser.open("http://127.0.0.1:5051")
    except Exception as e:
        loading_error = str(e)
        print(f"Error loading models: {e}")

# Start loading in background
threading.Thread(target=load_models_async, daemon=True).start()

# Helper for health check
@app.route('/api/health')
def health_check():
    return jsonify({
        "ready": models_ready,
        "error": loading_error,
        "device": device
    })

# OpenAI Compatibility Endpoints
@app.route('/v1/models', methods=['GET'])
def list_models():
    # Return a mock list of models to satisfy discovery
    voices = get_voices().get_json()
    model_list = []
    for v in voices:
        model_list.append({
            "id": v['id'],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "qwen3-tts"
        })
    return jsonify({"data": model_list})

@app.route('/v1/audio/speech', methods=['POST'])
def openai_speech():
    """OpenAI-compatible speech endpoint"""
    data = request.json
    text = data.get('input', '')
    voice_id = data.get('voice', 'vivian')
    speed = data.get('speed', 1.0)
    
    if not models_ready:
        return jsonify({"error": "Models are still loading"}), 503
    if not text:
        return jsonify({"error": "No input text"}), 400

    try:
        # Check if it's a saved voice
        is_custom = voice_id.startswith('custom_')
        custom_voice = None
        if is_custom:
            saved = load_saved_voices()
            custom_voice = next((v for v in saved if v['id'] == voice_id), None)
            
        with gpu_lock:
            if custom_voice:
                if custom_voice['type'] == 'design':
                    prepare_model('designer')
                    wavs, sr = voice_designer.generate_voice_design(text, instruct=custom_voice['prompt'], language='auto')
                else: # clone
                    prepare_model('tts')
                    wavs, sr = tts_base.generate_voice_clone(text, ref_audio=custom_voice['ref_path'], language='auto', x_vector_only_mode=True)
            else:
                prepare_model('tts')
                wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, language='auto')
                
        if wavs is not None:
            audio = wavs[0]
            temp_file = f"openai_{uuid.uuid4()}.wav"
            temp_path = os.path.join(OUTPUT_DIR, temp_file)
            sf.write(temp_path, audio, sr)
            
            # Return the file as expected by OpenAI clients
            return send_file(temp_path, mimetype="audio/wav")
        else:
            return jsonify({"error": "Generation failed"}), 500
    except Exception as e:
        print(f"Error in /v1/audio/speech: {e}")
        return jsonify({"error": str(e)}), 500

# Session storage and cleanup
SESSIONS = {}
SESSION_DIR = os.path.join(OUTPUT_DIR, "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

# Lock for GPU access to prevent concurrent generation issues
gpu_lock = threading.Lock()

def clear_vram(force=False):
    """Optimized memory cleanup for MPS/CUDA with synchronization."""
    import gc
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    print("--- VRAM Cleared ---", flush=True)

def move_to_device(model, target_device):
    """Safely move Qwen3TTSModel or underlying torch model to device."""
    if model is None: return
    try:
        # Check for model.model first (the underlying torch module in Qwen3TTSModel)
        if hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model.to(target_device)
        elif hasattr(model, 'to'):
            model.to(target_device)
        else:
            # print(f"Warning: Model type {type(model)} has no 'to' attribute and no '.model' attribute.")
            pass
    except Exception as e:
        print(f"Error moving {type(model)} to {target_device}: {e}")

def release_gpu_memory():
    """Move all models to CPU to free up MPS for MLX."""
    global tts_custom, tts_base, voice_designer
    print("--- RELEASING GPU memory (Torch -> CPU) for MLX path... ---", flush=True)
    try:
        move_to_device(tts_custom, "cpu")
        move_to_device(tts_base, "cpu")
        move_to_device(voice_designer, "cpu")
        clear_vram(force=True)
    except Exception as e:
        print(f"Warning during memory release: {e}", flush=True)

def acquire_gpu_memory(model_type='all'):
    """Move requested models back to target device."""
    global tts_custom, tts_base, voice_designer
    print(f"--- ACQUIRING GPU memory (CPU -> {device}) for {model_type}... ---", flush=True)
    try:
        if (model_type == 'all' or model_type == 'tts'):
            move_to_device(tts_base, device)
            move_to_device(tts_custom, device)
        if (model_type == 'all' or model_type == 'designer'):
            move_to_device(voice_designer, device)
        clear_vram()
    except Exception as e:
        print(f"Error during memory acquisition: {e}", flush=True)

def prepare_model(model_type):
    """Ensure models are on the correct device."""
    if not models_ready: return
    acquire_gpu_memory(model_type)

def split_text_streaming(text, first_chunk_len=400, target_len=900):
    """Balanced splitting for Mac: large enough for prosody, small enough for responsiveness."""
    # Split by common sentence delimiters
    parts = re.split(r'([.!?。！？\n])', text)
    
    sentence_list = []
    for i in range(0, len(parts)-1, 2):
        sentence_list.append(parts[i] + parts[i+1])
    if len(parts) % 2 == 1 and parts[-1]:
        sentence_list.append(parts[-1])
        
    chunks = []
    current_chunk = ""
    
    for sentence in sentence_list:
        sentence = sentence.strip()
        if not sentence: continue
        
        limit = first_chunk_len if not chunks else target_len
        
        if len(current_chunk) + len(sentence) < limit:
            current_chunk += sentence + " "
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    # If no delimiters found or chunks empty, just split by length
    if not chunks:
        for i in range(0, len(text), target_len):
            chunks.append(text[i:i+target_len])
            
    # Safety: if any chunk is still huge, force split it
    final_chunks = []
    for c in chunks:
        if len(c) > 1000:
            words = c.split(' ')
            temp = ""
            for w in words:
                if len(temp) + len(w) < target_len:
                    temp += w + " "
                else:
                    if temp: final_chunks.append(temp.strip())
                    temp = w + " "
            if temp: final_chunks.append(temp.strip())
        else:
            final_chunks.append(c)
            
    return [c for c in final_chunks if c.strip()]

def get_speed_instruction(speed, language='english'):
    """Maps numerical speed to natural language instructions."""
    if speed is None: return ""
    
    speed = float(speed)
    if 0.9 <= speed <= 1.1: return ""
    
    # Spanish mapping
    if language == 'spanish':
        if speed <= 0.6: return "habla muy despacio,"
        if speed <= 0.8: return "habla un poco más lento,"
        if speed >= 1.5: return "habla muy rápido,"
        if speed >= 1.2: return "habla un poco más rápido,"
    
    # Default English mapping
    if speed <= 0.6: return "speak very slowly,"
    if speed <= 0.8: return "speak a bit slower,"
    if speed >= 1.5: return "speak very fast,"
    if speed >= 1.2: return "speak a bit faster,"
    
    return ""

def generation_worker(session_id, chunks, voice_id, instruction, language, mode, extra_info, speed=1.0, engine='standard'):
    global tts_base, tts_custom, voice_designer
    session = SESSIONS[session_id]
    session_path = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    
    speed_ins = get_speed_instruction(speed, language)
    
    # Logic for Saved Voices
    is_custom = voice_id.startswith('custom_') if voice_id else False
    custom_voice = None
    if is_custom:
        saved = load_saved_voices()
        custom_voice = next((v for v in saved if v['id'] == voice_id), None)
        print(f"[{session_id[:8]}] Custom voice detected: {voice_id} -> Found? {custom_voice is not None}")
    
    # Pre-calculate voice clone prompt if needed
    cached_prompt = None
    if mode == 'clone' or (custom_voice and custom_voice.get('type') == 'clone'):
        ref_path = extra_info if mode == 'clone' else custom_voice['ref_path']
        if os.path.exists(ref_path):
            session['status'] = "Analizando voz de referencia..."
            print(f"[{session_id[:8]}] Pre-calculating voice clone prompt for {ref_path}...")
            prepare_model('tts')
            with gpu_lock:
                try:
                    # Use x_vector_only_mode=True to avoid requiring ref_text
                    cached_prompt = tts_base.create_voice_clone_prompt(ref_audio=ref_path, x_vector_only_mode=True)
                    session['status'] = "Iniciando lectura..."
                except Exception as e:
                    print(f"[{session_id[:8]}] Error pre-calculating prompt: {e}")
                    session['error'] = f"Prompt Error: {str(e)}"
                    session['status'] = "Error"
                    return

    for i, text in enumerate(chunks):
        if session.get('stopped'): break
        
        # Periodic Memory Cleanup
        if i > 0 and i % 5 == 0:
            clear_vram()
            
        session['status'] = f"Generating chunk {i+1}/{len(chunks)}..."
        print(f"[{session_id[:8]}] >>> Starting generation for chunk {i+1}/{len(chunks)}...")
        
        try:
            if engine == 'optimized':
                with gpu_lock:
                    release_gpu_memory()
                    from q3_tts_tool import generate_speech
                    filepath = os.path.join(session_path, f"chunk_{i}.wav")
                    
                    # Optimized Engine Path
                    if mode == 'design' or (custom_voice and custom_voice['type'] == 'design'):
                        ref_audio_path = os.path.join(session_path, "design_reference.wav")
                        
                        if not os.path.exists(ref_audio_path):
                            # PHASE 0: Design the voice DNA once
                            session['status'] = "Designing voice Identity (MLX)..."
                            design_prompt = (extra_info or "").strip() if mode == 'design' else custom_voice['prompt'].strip()
                            if speed != 1.0:
                                speed_ins = get_speed_instruction(speed, language)
                                design_prompt = f"{speed_ins} {design_prompt}"
                            
                            design_snippet = ' '.join(chunks[0].split()[:15])
                            print(f"[{session_id[:8]}] Designing Identity (MLX) with: {design_snippet[:50]}...", flush=True)
                            
                            res = generate_speech(design_snippet, instruction=design_prompt, output_path=ref_audio_path)
                            if "Success" not in res:
                                raise ValueError(f"Optimized Identity Design failed: {res}")
                        
                        session['status'] = f"Generating chunk {i+1}/{len(chunks)} (MLX Consistent)..."
                        res = generate_speech(text, clone_path=ref_audio_path, output_path=filepath)
                        if "Success" not in res:
                            raise ValueError(f"Optimized Consistent Clone failed: {res}")
                    
                    elif mode == 'clone' or (custom_voice and custom_voice['type'] == 'clone'):
                        ref = extra_info if mode == 'clone' else custom_voice['ref_path']
                        res = generate_speech(text, clone_path=ref, output_path=filepath)
                        if "Success" not in res:
                            raise ValueError(f"Optimized Clone failed: {res}")
                    
                    else: # Preset
                        res = generate_speech(text, instruction=instruction, output_path=filepath)
                        if "Success" not in res:
                            raise ValueError(f"Optimized Preset failed: {res}")
                
                    wavs = [True] # Dummy
                    sr = 24000
            else:
                # Standard Torch Path
                with gpu_lock:
                    if mode == 'design' or (custom_voice and custom_voice['type'] == 'design'):
                        ref_audio_path = os.path.join(session_path, "design_reference.wav")
                        if not os.path.exists(ref_audio_path):
                            session['status'] = "Diseñando identidad vocal..."
                            prompt = (extra_info or "").strip() if mode == 'design' else custom_voice['prompt'].strip()
                            if speed_ins:
                                prompt = f"{speed_ins} {prompt}"
                            
                            design_snippet = ' '.join(chunks[0].split()[:15])
                            print(f"[{session_id[:8]}] Designing Identity (Torch) with: {design_snippet[:50]}...", flush=True)
                            prepare_model('designer')
                            wavs_ref, sr_ref = voice_designer.generate_voice_design(design_snippet, instruct=prompt.strip(), language=language)
                            if wavs_ref is not None:
                                sf.write(ref_audio_path, wavs_ref[0], sr_ref)
                            else:
                                raise ValueError("Torch Voice Identity Design failed")
                        
                        prepare_model('tts')
                        wavs, sr = tts_base.generate_voice_clone(text, ref_audio=ref_audio_path, language=language, x_vector_only_mode=True)
                            
                    elif mode == 'clone' or (custom_voice and custom_voice['type'] == 'clone'):
                        prepare_model('tts')
                        if cached_prompt:
                            wavs, sr = tts_base.generate_voice_clone(text, voice_clone_prompt=cached_prompt, language=language)
                        else:
                            ref = extra_info if mode == 'clone' else custom_voice['ref_path']
                            wavs, sr = tts_base.generate_voice_clone(text, ref_audio=ref, language=language, x_vector_only_mode=True)
                            
                    else: # Preset/Standard
                        prepare_model('tts')
                        full_instruction = instruction
                        if speed_ins:
                            full_instruction = f"{speed_ins} {instruction}" if instruction else speed_ins.strip(',')
                        
                        if full_instruction:
                            wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, instruct=full_instruction, language=language)
                        else:
                            wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, language=language)
                    
            if wavs is not None:
                if engine != 'optimized':
                    audio = wavs[0]
                    # APPLY SOFT FADE (5ms) to prevent clicks between chunks
                    fade_len = int(sr * 0.005)
                    if len(audio) > fade_len * 2:
                        fade_curve = np.linspace(0, 1, fade_len)
                        audio[:fade_len] *= fade_curve
                        audio[-fade_len:] *= fade_curve[::-1]
                        
                    filepath = os.path.join(session_path, f"chunk_{i}.wav")
                    sf.write(filepath, audio, sr)
                
                session['ready_chunks'].append(i)
                progress_pct = int((len(session['ready_chunks']) / len(chunks)) * 100)
                session['status'] = f"Processing {len(session['ready_chunks'])}/{len(chunks)} ({progress_pct}%)"
                print(f"[{session_id[:8]}] Progress: {len(session['ready_chunks'])}/{len(chunks)} - Chunk {i} saved.")
            else:
                raise ValueError("Generation failed to return audio")
                    
        except Exception as e:
            session['error'] = str(e)
            print(f"[{session_id[:8]}] ERROR in worker: {e}")
            import traceback
            traceback.print_exc()
            break
            
    if not session.get('error'):
        session['status'] = "Completed"
        print(f"[{session_id[:8]}] Generation Completed.")
    else:
        session['status'] = "Error"

@app.route('/api/stream/start', methods=['POST'])
def stream_start():
    data = request.json
    text = data.get('text', '')
    voice_id = data.get('voice_id', 'vivian')
    instruction = data.get('instruction', '')
    lang_code = data.get('language', 'zh')
    language = LANGUAGE_MAP.get(lang_code, 'english')
    mode = data.get('mode', 'preset')
    extra_info = data.get('extra_info', '') # ref_path or description
    speed = data.get('speed', 1.0)
    engine = data.get('engine', 'standard')
    
    if not models_ready and engine == 'standard':
        return jsonify({"error": "Models are still loading. Please wait."}), 503
    
    if not text:
        return jsonify({"error": "No text"}), 400
    
    session_id = str(uuid.uuid4())
    chunks = split_text_streaming(text)
    
    SESSIONS[session_id] = {
        "id": session_id,
        "total_chunks": len(chunks),
        "ready_chunks": [],
        "status": "Starting",
        "error": None
    }
    
    thread = threading.Thread(target=generation_worker, args=(session_id, chunks, voice_id, instruction, language, mode, extra_info, speed, engine))
    thread.start()
    
    return jsonify({"session_id": session_id, "total_chunks": len(chunks)})
    
@app.route('/api/stream/start-clone', methods=['POST'])
def stream_start_clone():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400
    
    audio_file = request.files['audio']
    text = request.form.get('text', '')
    lang_code = request.form.get('language', 'zh')
    language = LANGUAGE_MAP.get(lang_code, 'english')
    speed = float(request.form.get('speed', 1.0))
    
    if not models_ready:
        return jsonify({"error": "Models are still loading"}), 503
        
    session_id = str(uuid.uuid4())
    ref_path = os.path.join(SESSION_DIR, f"ref_{session_id}.wav")
    audio_file.save(ref_path)
    
    chunks = split_text_streaming(text)
    SESSIONS[session_id] = {
        "id": session_id,
        "total_chunks": len(chunks),
        "ready_chunks": [],
        "status": "Initializing Clone...",
        "error": None
    }
    
    thread = threading.Thread(target=generation_worker, args=(session_id, chunks, None, "", language, 'clone', ref_path, speed))
    thread.start()
    
    return jsonify({"session_id": session_id, "total_chunks": len(chunks)})

@app.route('/api/stream/status/<session_id>', methods=['GET'])
def stream_status(session_id):
    if session_id not in SESSIONS:
        return jsonify({"error": "Session not found"}), 404
    return jsonify(SESSIONS[session_id])

@app.route('/api/stream/audio/<session_id>/<int:chunk_id>', methods=['GET'])
def stream_audio(session_id, chunk_id):
    filepath = os.path.join(SESSION_DIR, session_id, f"chunk_{chunk_id}.wav")
    if os.path.exists(filepath):
        return send_file(filepath, mimetype="audio/wav")
    return jsonify({"error": "Chunk not ready"}), 404

@app.route('/api/stream/stop/<session_id>', methods=['POST'])
def stream_stop(session_id):
    if session_id in SESSIONS:
        SESSIONS[session_id]['stopped'] = True
        return jsonify({"status": "Stopping..."})
    return jsonify({"error": "Session not found"}), 404

@app.route('/api/voices', methods=['GET'])
def get_voices():
    # Built-in presets
    presets = [
        {"id": "vivian", "name": "Vivian (Premium)", "type": "preset"},
        {"id": "ryan", "name": "Ryan (Premium)", "type": "preset"},
        {"id": "sohee", "name": "Sohee (Premium)", "type": "preset"},
        {"id": "aiden", "name": "Aiden (Premium)", "type": "preset"}
    ]
    # Saved voices
    saved = load_saved_voices()
    return jsonify(presets + saved)

@app.route('/api/save-voice', methods=['POST'])
def api_save_voice():
    data = request.json
    save_voice_to_json(data)
    return jsonify({"status": "success"})

@app.route('/api/extract-pdf', methods=['POST'])
def extract_pdf():
    if 'pdf' not in request.files:
        return jsonify({"error": "No file"}), 400
    pdf_file = request.files['pdf']
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return jsonify({"text": text})

@app.route('/')
def index():
    return send_file('index.html')

if __name__ == '__main__':
    # Use port 5051 as configured in previous versions
    app.run(host='0.0.0.0', port=5051, debug=False)
