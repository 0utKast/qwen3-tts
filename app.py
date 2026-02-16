import os
# Qwen3-TTS Studio v1.1.0
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
    print("\n--- CUDA Detected (NVIDIA) ---")
elif torch.backends.mps.is_available():
    device = "mps"
    print("\n--- MPS Detected (Apple Silicon) ---")
else:
    print("\n--- CPU Only Detected ---")
    print("WARNING: Running on CPU will be slow.")

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
            dtype=torch.float16 if device == "mps" else torch.bfloat16
        )
        print(f"--- Custom model loaded to {device} ---")
        
        print(f"--- Loading model {BASE_MODEL_ID}... ---")
        tts_base = Qwen3TTSModel.from_pretrained(
            BASE_MODEL_ID, 
            device_map=device, 
            dtype=torch.float16 if device == "mps" else torch.bfloat16
        )
        print(f"--- Base model loaded to {device} ---")
        
        print(f"--- Loading model {DESIGN_MODEL_ID}... ---")
        voice_designer = Qwen3TTSModel.from_pretrained(
            DESIGN_MODEL_ID, 
            device_map=device, 
            dtype=torch.float16 if device == "mps" else torch.bfloat16
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
    except Exception as e:
        loading_error = str(e)
        print(f"Error loading models: {e}")

# Start loading in background
threading.Thread(target=load_models_async, daemon=True).start()

print(f"--- Init: Using device {device} ---")

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
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif device == "mps":
        # Synchronize before clearing cache to ensure no operations are pending
        # This helps prevent stalls and deadlocks on Apple Silicon
        torch.mps.synchronize()
        torch.mps.empty_cache()

def move_to_device(model, target_device):
    """Safely move model to device, checking wrapper and underlying model."""
    if model is None: return
    try:
        if hasattr(model, 'to'):
            model.to(target_device)
        elif hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model.to(target_device)
    except Exception as e:
        print(f"Warning: Could not move model to {target_device}: {e}")

def prepare_model(model_type):
    """Placeholder for consistency, all three variants stay on Device."""
    pass

def split_text_streaming(text, first_chunk_len=400, target_len=900):
    """Balanced splitting for Mac: large enough for prosody, small enough for responsiveness."""
    # Split by common sentence delimiters
    parts = re.split(r'([.!?。！？\n])', text)
    
    chunks = []
    current_chunk = ""
    
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
        
    # If no delimiters found, just split by length
    if not chunks:
        for i in range(0, len(text), target_len):
            chunks.append(text[i:i+target_len])
            
    # Safety: if any chunk is still huge, force split it
    final_chunks = []
    for c in chunks:
        if len(c) > 600:
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

def generation_worker(session_id, chunks, voice_id, instruction, language, mode, extra_info, speed=1.0):
    global tts_base, tts_custom, voice_designer
    session = SESSIONS[session_id]
    session_path = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    
    speed_ins = get_speed_instruction(speed, language)
    
    session['status'] = f"Generating chunk 1/{len(chunks)}..."
    
    # Logic for Saved Voices - Move up to avoid UnboundLocalError
    is_custom = voice_id.startswith('custom_') if voice_id else False
    custom_voice = None
    if is_custom:
        saved = load_saved_voices()
        custom_voice = next((v for v in saved if v['id'] == voice_id), None)
        print(f"[{session_id[:8]}] Custom voice detected: {voice_id} -> Found? {custom_voice is not None}")
        if custom_voice:
            print(f"[{session_id[:8]}] Custom voice type: {custom_voice['type']}")
    
    # Pre-calculate voice clone prompt if needed
    cached_prompt = None
    if mode == 'clone' or (custom_voice and custom_voice.get('type') == 'clone'):
        ref_path = extra_info if mode == 'clone' else custom_voice['ref_path']
        if os.path.exists(ref_path):
            session['status'] = "Analizando voz de referencia (esto tarda unos segundos)..."
            print(f"[{session_id[:8]}] Pre-calculating voice clone prompt for {ref_path}...")
            prepare_model('tts')
            with gpu_lock:
                try:
                    # We use x_vector_only_mode=True to avoid requiring ref_text (ICL mode)
                    cached_prompt = tts_base.create_voice_clone_prompt(ref_audio=ref_path, x_vector_only_mode=True)
                    session['status'] = "Voz analizada. Iniciando lectura..."
                except Exception as e:
                    print(f"[{session_id[:8]}] Error pre-calculating prompt: {e}")
                    session['error'] = f"Prompt Error: {str(e)}"
                    session['status'] = "Error"
                    return

    for i, text in enumerate(chunks):
        if session.get('stopped'): break
        
        # Periodic Memory Cleanup every 5 chunks to prevent VRAM accumulation
        if i > 0 and i % 5 == 0:
            print(f"[{session_id[:8]}] Periodic VRAM cleanup...")
            clear_vram()
            
        session['status'] = f"Generating chunk {i+1}/{len(chunks)}..."
        print(f"[{session_id[:8]}] >>> Starting generation for chunk {i+1}/{len(chunks)}...")
        print(f"[{session_id[:8]}] Chunk text preview: {text[:50]}...")
        start_time = time.time()
        
        try:
            with gpu_lock:
                print(f"[{session_id[:8]}] GPU Lock Acquired. Current Time: {time.strftime('%H:%M:%S')}")
                if mode == 'design' or (custom_voice and custom_voice['type'] == 'design'):
                    current_mode = 'design'
                    prompt = extra_info if mode == 'design' else custom_voice['prompt']
                    if speed_ins:
                        prompt = f"{speed_ins} {prompt}"
                    prepare_model('designer')
                    wavs, sr = voice_designer.generate_voice_design(text, instruct=prompt, language=language)
                    
                elif mode == 'clone' or (custom_voice and custom_voice['type'] == 'clone'):
                    current_mode = 'clone'
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
                
                print(f"[{session_id[:8]}] Model execution completed in {time.time() - start_time:.2f}s.")

            if wavs is not None:
                gen_time = time.time() - start_time
                audio = wavs[0]
                
                # APPLY SOFT FADE (5ms) to prevent clicks between chunks
                # 5ms = samplerate * 0.005
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
                print(f"[{session_id[:8]}] Chunk {i} saved in {gen_time:.2f}s.")
                
                # Cleanup VRAM after each chunk
                clear_vram()
            else:
                raise ValueError("Generation failed to return audio")
                    
        except Exception as e:
            session['error'] = str(e)
            print(f"[{session_id[:8]}] ERROR in worker: {e}")
            import traceback
            traceback.print_exc()
            break
            
    # Keep models on GPU for high-performance RTX 3090
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
    
    if not models_ready:
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
    
    thread = threading.Thread(target=generation_worker, args=(session_id, chunks, voice_id, instruction, language, mode, extra_info, speed))
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
        return send_file(filepath)
    return jsonify({"error": "Chunk not ready"}), 404

@app.route('/api/stream/concatenate/<session_id>', methods=['GET'])
def stream_concatenate(session_id):
    if session_id not in SESSIONS:
        return jsonify({"error": "Session not found"}), 404
    
    session = SESSIONS[session_id]
    if session['status'] != 'Completed':
        return jsonify({"error": "Generation not completed"}), 400
    
    try:
        session_path = os.path.join(SESSION_DIR, session_id)
        all_audio = []
        sample_rate = None
        
        # Sort chunks to ensure correct order
        for i in range(session['total_chunks']):
            chunk_path = os.path.join(session_path, f"chunk_{i}.wav")
            if os.path.exists(chunk_path):
                data, sr = sf.read(chunk_path)
                all_audio.append(data)
                sample_rate = sr
            else:
                return jsonify({"error": f"Chunk {i} missing"}), 500
        
        if not all_audio:
            return jsonify({"error": "No audio chunks found"}), 500
            
        # Concatenate and save final file
        final_audio = np.concatenate(all_audio)
        final_filename = f"final_{session_id}.wav"
        final_path = os.path.join(OUTPUT_DIR, final_filename)
        sf.write(final_path, final_audio, sample_rate)
        
        return jsonify({"url": f"/static/audio/{final_filename}"})
    except Exception as e:
        print(f"Error concatenating chunks: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/extract-text', methods=['POST'])
def extract_pdf_text():
    print("--- Received PDF extraction request ---")
    if 'file' not in request.files:
        print("Error: No file in request")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    if file.filename == '':
        print("Error: Empty filename")
        return jsonify({"error": "No selected file"}), 400

    try:
        content = file.read()
        print(f"Reading PDF: {file.filename} ({len(content)} bytes)")
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        
        extracted_text = text.strip()
        print(f"Extracted {len(extracted_text)} characters.")
        return jsonify({"text": extracted_text})
    except Exception as e:
        print(f"PDF Error: {str(e)}")
        return jsonify({"error": f"PDF Extraction Error: {str(e)}"}), 500

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/style.css')
def style():
    return send_file('style.css')

@app.route('/script.js')
def script():
    return send_file('script.js')

@app.route('/api/voices', methods=['GET'])
def get_voices():
    # Predefined premium voices
    voices = [
        {"id": "vivian", "name": "Vivian", "description": "Young female, bright, energetic", "gender": "Female", "type": "preset"},
        {"id": "ryan", "name": "Ryan", "description": "Dynamic male, clear rhythm", "gender": "Male", "type": "preset"},
        {"id": "aiden", "name": "Aiden", "description": "Sunny young male, US accent", "gender": "Male", "type": "preset"},
        {"id": "sohee", "name": "Sohee", "description": "Warm and kind female, Korean accent", "gender": "Female", "type": "preset"}
    ]
    # Add saved voices
    saved = load_saved_voices()
    return jsonify(voices + saved)

@app.route('/api/voices/save', methods=['POST'])
def save_voice_endpoint():
    # Handle both JSON and Multipart (for cloning)
    if request.is_json:
        data = request.json
        name = data.get('name')
        description = data.get('description', '')
        voice_type = data.get('type')
        value = data.get('value')
        audio_file = None
    else:
        name = request.form.get('name')
        description = request.form.get('description', '')
        voice_type = request.form.get('type')
        value = request.form.get('value')
        audio_file = request.files.get('audio')

    if not name or not voice_type:
        return jsonify({"error": "Name and Type are required"}), 400
    
    voice_id = f"custom_{uuid.uuid4().hex[:8]}"
    new_voice = {
        "id": voice_id,
        "name": name,
        "type": voice_type,
        "description": description or f"Custom {voice_type} voice",
        "gender": "Custom"
    }
    
    if voice_type == 'design':
        if not value: return jsonify({"error": "Design prompt required"}), 400
        new_voice["prompt"] = value
    elif voice_type == 'clone':
        # Prioritize uploaded file, fall back to value (path)
        perm_filename = f"{voice_id}.wav"
        perm_path = os.path.join(CUSTOM_VOICES_DIR, perm_filename)
        
        if audio_file:
            audio_file.save(perm_path)
            new_voice["ref_path"] = perm_path
        elif value and os.path.exists(value):
            shutil.copy(value, perm_path)
            new_voice["ref_path"] = perm_path
        else:
            return jsonify({"error": "Reference audio required for cloning"}), 400
            
    save_voice_to_json(new_voice)
    print(f"--- Voice Saved: {name} ({voice_id}) ---")
    return jsonify({"success": True, "voice": new_voice})

@app.route('/api/voices/delete/<voice_id>', methods=['POST'])
def delete_voice_endpoint(voice_id):
    voices = load_saved_voices()
    voice_to_delete = next((v for v in voices if v['id'] == voice_id), None)
    
    if not voice_to_delete:
        return jsonify({"error": "Voice not found"}), 404
        
    # Delete associated file if it's a clone
    if voice_to_delete.get('type') == 'clone':
        ref_path = voice_to_delete.get('ref_path')
        if ref_path and os.path.exists(ref_path):
            try:
                os.remove(ref_path)
                print(f"--- Deleted ref file: {ref_path} ---")
            except Exception as e:
                print(f"Error deleting file {ref_path}: {e}")
                
    # Remove from list
    new_voices = [v for v in voices if v['id'] != voice_id]
    with open(SAVED_VOICES_FILE, 'w', encoding='utf-8') as f:
        json.dump(new_voices, f, ensure_ascii=False, indent=2)
        
    print(f"--- Voice Deleted: {voice_to_delete.get('name')} ({voice_id}) ---")
    return jsonify({"success": True})

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', '')
    voice_id = data.get('voice_id', 'vivian')
    instruction = data.get('instruction', '')
    lang_code = data.get('language', 'zh')
    speed = data.get('speed', 1.0)
    language = LANGUAGE_MAP.get(lang_code, 'english') # Map or fallback to english
    
    if not models_ready:
        return jsonify({"error": "Models are still loading. Please wait."}), 503
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if tts_custom is None or tts_base is None:
        return jsonify({"error": "TTS models not loaded properly. Check terminal for Hardware Acceleration/Memory errors."}), 503
    
    try:
        # Check if optimized engine is requested
        engine = data.get('engine', 'standard')
        
        if engine == 'optimized':
            from q3_tts_tool import generate_speech
            filename = f"opt_{uuid.uuid4()}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            
            # Prepare instruction for Simon's script
            voice_ins = instruction
            if speed != 1.0:
                speed_ins = get_speed_instruction(speed, language)
                voice_ins = f"{speed_ins} {instruction}" if instruction else speed_ins.strip(',')

            # For now, we only support presets/descriptions in optimized mode via this path
            res = generate_speech(text, instruction=voice_ins, output_path=filepath)
            
            if "Success" in res:
                return jsonify({"url": f"/static/audio/{filename}"})
            else:
                return jsonify({"error": f"Optimized engine failed: {res}"}), 500

        # Check if it's a saved voice
        is_custom = voice_id.startswith('custom_')
        custom_voice = None
        if is_custom:
            saved = load_saved_voices()
            custom_voice = next((v for v in saved if v['id'] == voice_id), None)
            
        if custom_voice:
            if custom_voice['type'] == 'design':
                prepare_model('designer')
                prompt = custom_voice['prompt']
                speed_ins = get_speed_instruction(speed, language)
                if speed_ins:
                    prompt = f"{speed_ins} {prompt}"
                    
                with gpu_lock:
                    wavs, sr = voice_designer.generate_voice_design(text, instruct=prompt, language=language)
            else: # clone
                prepare_model('tts')
                with gpu_lock:
                    wavs, sr = tts_base.generate_voice_clone(text, ref_audio=custom_voice['ref_path'], language=language, x_vector_only_mode=True)
        else:
            # Preset Logic
            prepare_model('tts')
            speed_ins = get_speed_instruction(speed, language)
            
            with gpu_lock:
                full_instruction = instruction
                if speed_ins:
                    full_instruction = f"{speed_ins} {instruction}" if instruction else speed_ins.strip(',')
                
                if full_instruction:
                    try:
                        wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, instruct=full_instruction, language=language)
                    except Exception as e:
                        if "does not support generate_custom_voice" in str(e):
                            return jsonify({"error": "The loaded model does not support presets. Please ensure MODEL_ID is set to 'CustomVoice' in app.py."}), 400
                        else:
                            raise e
                else:
                    wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, language=language)
                
        if wavs is not None:
            audio = wavs[0]
            filename = f"{uuid.uuid4()}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            sf.write(filepath, audio, sr)
            
            # Limpiar memoria tras generar
            clear_vram(force=True)
            
            return jsonify({"url": f"/static/audio/{filename}"})
        else:
            raise ValueError("Generation failed to return audio")
    except Exception as e:
        print(f"Error in /api/generate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/clone', methods=['POST'])
def clone_voice():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file uploaded"}), 400
    
    audio_file = request.files['audio']
    text = request.form.get('text', '')
    lang_code = request.form.get('language', 'zh')
    language = LANGUAGE_MAP.get(lang_code, 'english')
    
    if not models_ready:
        return jsonify({"error": "Models are still loading. Please wait."}), 503
    
    if audio_file is None:
        return jsonify({"error": "Base model (tts_base) not loaded for cloning."}), 503
    try:
        ref_path = "temp_ref.wav"
        audio_file.save(ref_path)
        
        prepare_model('tts')
        with gpu_lock:
            wavs, sr = tts_base.generate_voice_clone(text, ref_audio=ref_path, language=language, x_vector_only_mode=True)
            
        if wavs is not None and len(wavs) > 0:
            audio = wavs[0]
            filename = f"cloned_{uuid.uuid4()}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            sf.write(filepath, audio, sr)
            return jsonify({"url": f"/static/audio/{filename}"})
        else:
            raise ValueError("Cloning failed to return audio")
    except Exception as e:
        print(f"Error in /api/clone: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/design', methods=['POST'])
def design_voice():
    data = request.json
    text = data.get('text', '')
    description = data.get('description', '')
    lang_code = data.get('language', 'zh')
    speed = data.get('speed', 1.0)
    language = LANGUAGE_MAP.get(lang_code, 'english')
    
    if not text or not description:
        return jsonify({"error": "Text and description required"}), 400
    
    if voice_designer is None:
        return jsonify({"error": "Model 'voice_designer' not loaded."}), 503
    
    try:
        prepare_model('designer')
        speed_ins = get_speed_instruction(speed, language)
        prompt = description
        if speed_ins:
            prompt = f"{speed_ins} {description}"
            
        with gpu_lock:
            wavs, sr = voice_designer.generate_voice_design(text, instruct=prompt, language=language)
            
        if wavs is not None and len(wavs) > 0:
            audio = wavs[0]
            filename = f"designed_{uuid.uuid4()}.wav"
            filepath = os.path.join(OUTPUT_DIR, filename)
            sf.write(filepath, audio, sr)
            return jsonify({"url": f"/static/audio/{filename}"})
        else:
            raise ValueError("Design failed to return audio")
    except Exception as e:
        print(f"Error in /api/design: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # use_reloader=False is CRITICAL here to avoid loading models twice
    app.run(host='0.0.0.0', port=5051, debug=False)
