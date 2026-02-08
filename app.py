import os
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

# Configuration
CUSTOM_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
BASE_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DESIGN_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUTPUT_DIR = "static/audio"
SAVED_VOICES_FILE = "voices.json"
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

# Check for CUDA availability
if not torch.cuda.is_available():
    print("\n" + "!"*50)
    print("ERROR: CUDA is not available to PyTorch.")
    print("The models REQUIRE a GPU with CUDA to run efficiently.")
    print("Please reinstall PyTorch with CUDA support using:")
    print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall")
    print("!"*50 + "\n")

# Initialize models
tts_custom = None
tts_base = None
voice_designer = None

print(f"--- Init: Checking GPU and RAM ---")
# User has an RTX 3090 with 24GB VRAM. 
# We'll load models without restrictive limits to avoid device-mismatch fragmentation.

try:
    print(f"--- Loading Custom tts model ({CUSTOM_MODEL_ID})... ---")
    tts_custom = Qwen3TTSModel.from_pretrained(
        CUSTOM_MODEL_ID, 
        device_map="cuda", 
        dtype=torch.bfloat16
    )
    print("--- Custom model loaded to GPU ---")
    
    print(f"--- Loading Base tts model ({BASE_MODEL_ID})... ---")
    tts_base = Qwen3TTSModel.from_pretrained(
        BASE_MODEL_ID, 
        device_map="cuda", 
        dtype=torch.bfloat16
    )
    print("--- Base model loaded to GPU ---")
    
    print(f"--- Loading Voice Designer model ({DESIGN_MODEL_ID})... ---")
    voice_designer = Qwen3TTSModel.from_pretrained(
        DESIGN_MODEL_ID, 
        device_map="cuda", 
        dtype=torch.bfloat16
    )
    print("--- Voice Designer loaded to GPU ---")
    
    # Open browser automatically once ready
    webbrowser.open("http://127.0.0.1:5000")
except Exception as e:
    print(f"Error loading models: {e}")

# Session storage and cleanup
SESSIONS = {}
SESSION_DIR = os.path.join(OUTPUT_DIR, "sessions")
os.makedirs(SESSION_DIR, exist_ok=True)

# Lock for GPU access to prevent concurrent generation issues
gpu_lock = threading.Lock()

def clear_vram():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

def move_to_device(model, device):
    """Safely move model to device, checking wrapper and underlying model."""
    if model is None: return
    try:
        if hasattr(model, 'to'):
            model.to(device)
        elif hasattr(model, 'model') and hasattr(model.model, 'to'):
            model.model.to(device)
    except Exception as e:
        print(f"Warning: Could not move model to {device}: {e}")

def prepare_model(model_type):
    """Placeholder for consistency, all three variants stay on GPU."""
    pass

def split_text_streaming(text, first_chunk_len=1000, target_len=2000):
    """Asymmetric splitting for faster initial response."""
    # Simple split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
    chunks = []
    current_chunk = ""
    
    for p in paragraphs:
        limit = first_chunk_len if not chunks else target_len
        if len(current_chunk) + len(p) < limit:
            current_chunk += p + "\n"
        else:
            if current_chunk: chunks.append(current_chunk.strip())
            current_chunk = p + "\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # If a chunk is still too long, split it by sentences (coarse)
    refined_chunks = []
    for c in chunks:
        limit = first_chunk_len if not refined_chunks else target_len
        if len(c) > limit + 500:
            sentences = re.split(r'([.!?])\s+', c)
            temp = ""
            for i in range(0, len(sentences), 2):
                s = sentences[i] + (sentences[i+1] if i+1 < len(sentences) else "")
                if len(temp) + len(s) < limit:
                    temp += s + " "
                else:
                    if temp: refined_chunks.append(temp.strip())
                    temp = s + " "
            if temp: refined_chunks.append(temp.strip())
        else:
            refined_chunks.append(c)
            
    return refined_chunks

def generation_worker(session_id, chunks, voice_id, instruction, language, mode, extra_info):
    session = SESSIONS[session_id]
    session_path = os.path.join(SESSION_DIR, session_id)
    os.makedirs(session_path, exist_ok=True)
    
    for i, text in enumerate(chunks):
        if session.get('stopped'): break
        
        try:
                # Logic for Saved Voices
                is_custom = voice_id.startswith('custom_')
                custom_voice = None
                if is_custom:
                    saved = load_saved_voices()
                    custom_voice = next((v for v in saved if v['id'] == voice_id), None)
                
                if mode == 'design' or (custom_voice and custom_voice['type'] == 'design'):
                    current_mode = 'design'
                    prompt = extra_info if mode == 'design' else custom_voice['prompt']
                    prepare_model('designer')
                    with gpu_lock:
                        wavs, sr = voice_designer.generate_voice_design(text, instruct=prompt, language=language)
                elif mode == 'clone' or (custom_voice and custom_voice['type'] == 'clone'):
                    current_mode = 'clone'
                    ref = extra_info if mode == 'clone' else custom_voice['ref_path']
                    prepare_model('tts')
                    with gpu_lock:
                        wavs, sr = tts_base.generate_voice_clone(text, ref_audio=ref, language=language, x_vector_only_mode=True)
                else: # Preset/Standard
                    prepare_model('tts')
                    with gpu_lock:
                        try:
                            if instruction:
                                wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, instruct=instruction, language=language)
                            else:
                                wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, language=language)
                        except Exception as e:
                            # If for some reason it fails, we keep the error
                            raise e

                if wavs is not None:
                    audio = wavs[0]
                    filepath = os.path.join(session_path, f"chunk_{i}.wav")
                    sf.write(filepath, audio, sr)
                    
                    session['ready_chunks'].append(i)
                    session['status'] = f"Processing {i+1}/{len(chunks)}"
                else:
                    raise ValueError("Generation failed to return audio")
                    
        except Exception as e:
            session['error'] = str(e)
            print(f"Error in worker: {e}")
            break
            
    # Keep models on GPU for high-performance RTX 3090
    session['status'] = "Completed" if not session.get('error') else "Error"

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
    
    thread = threading.Thread(target=generation_worker, args=(session_id, chunks, voice_id, instruction, language, mode, extra_info))
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

@app.route('/api/extract-text', methods=['POST'])
def extract_pdf_text():
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400
    
    file = request.files['file']
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text() + "\n"
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

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
    data = request.json
    name = data.get('name')
    description = data.get('description', '')
    voice_type = data.get('type') # 'design' or 'clone'
    value = data.get('value') # description or temp_filename
    
    if not name or not voice_type or not value:
        return jsonify({"error": "Name, type and value required"}), 400
    
    voice_id = f"custom_{uuid.uuid4().hex[:8]}"
    new_voice = {
        "id": voice_id,
        "name": name,
        "type": voice_type,
        "description": description or f"Custom {voice_type} voice",
        "gender": "Custom"
    }
    
    if voice_type == 'design':
        new_voice["prompt"] = value
    elif voice_type == 'clone':
        # Move temp file to permanent storage
        perm_filename = f"{voice_id}.wav"
        perm_path = os.path.join(CUSTOM_VOICES_DIR, perm_filename)
        if os.path.exists(value):
            shutil.copy(value, perm_path)
            new_voice["ref_path"] = perm_path
        else:
            return jsonify({"error": "Reference audio not found"}), 400
            
    save_voice_to_json(new_voice)
    return jsonify({"success": True, "voice": new_voice})

@app.route('/api/generate', methods=['POST'])
def generate():
    data = request.json
    text = data.get('text', '')
    voice_id = data.get('voice_id', 'vivian')
    instruction = data.get('instruction', '')
    lang_code = data.get('language', 'zh')
    language = LANGUAGE_MAP.get(lang_code, 'english') # Map or fallback to english
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    if tts_custom is None or tts_base is None:
        return jsonify({"error": "TTS models not loaded properly. Check terminal for CUDA/Memory errors."}), 503
    
    try:
        # Check if it's a saved voice
        is_custom = voice_id.startswith('custom_')
        custom_voice = None
        if is_custom:
            saved = load_saved_voices()
            custom_voice = next((v for v in saved if v['id'] == voice_id), None)
            
        if custom_voice:
            if custom_voice['type'] == 'design':
                prepare_model('designer')
                with gpu_lock:
                    wavs, sr = voice_designer.generate_voice_design(text, instruct=custom_voice['prompt'], language=language)
            else: # clone
                prepare_model('tts')
                with gpu_lock:
                    wavs, sr = tts_base.generate_voice_clone(text, ref_audio=custom_voice['ref_path'], language=language, x_vector_only_mode=True)
        else:
            # Preset Logic
            prepare_model('tts')
            with gpu_lock:
                if instruction:
                    try:
                        wavs, sr = tts_custom.generate_custom_voice(text, speaker=voice_id, instruct=instruction, language=language)
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
    
    if tts_base is None:
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
    language = LANGUAGE_MAP.get(lang_code, 'english')
    
    if not text or not description:
        return jsonify({"error": "Text and description required"}), 400
    
    if voice_designer is None:
        return jsonify({"error": "Model 'voice_designer' not loaded."}), 503
    
    try:
        prepare_model('designer')
        with gpu_lock:
            wavs, sr = voice_designer.generate_voice_design(text, instruct=description, language=language)
            
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
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
