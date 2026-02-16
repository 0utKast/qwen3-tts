import subprocess
import os
import sys
import threading
import queue
import time
from pathlib import Path

def generate_speech(text, instruction=None, clone_path=None, output_path="output.wav"):
    """
    Generates speech using Qwen3-TTS via the optimized Simon Willison script and uv.
    
    Args:
        text (str): The text to convert to speech.
        instruction (str, optional): Voice description (e.g., 'calm female voice').
        clone_path (str, optional): Path to a 3-second audio sample for voice cloning.
        output_path (str, optional): Path where the output wav file will be saved.
    """
    script_path = os.path.join(os.path.dirname(__file__), "q3_tts_local.py")
    if not os.path.exists(script_path):
        return f"Error: Local script '{script_path}' not found. Please ensure q3_tts_local.py is in the same directory."

    # Click expects options BEFORE arguments
    cmd = ["uv", "run", str(script_path), "--"]
    
    # Add options first
    cmd.extend(["-o", str(output_path), "-v"])
    if instruction:
        cmd.extend(["-i", str(instruction)])
    if clone_path:
        if not os.path.exists(clone_path):
            return f"Error: Clone path '{clone_path}' does not exist."
        cmd.extend(["--clone", str(clone_path)])
    
    # Add the text argument LAST
    cmd.append(str(text))
    
    # Cast everything to string for safety
    cmd = [str(c) for c in cmd]
    print(f"Executing: {' '.join(cmd)}", flush=True)
    
    try:
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Thread-safe queue for process output
        output_queue = queue.Queue()
        
        def stream_reader(pipe, q):
            try:
                for line in iter(pipe.readline, ''):
                    q.put(line)
                pipe.close()
            except Exception:
                pass

        # Start background thread to read output
        reader_thread = threading.Thread(target=stream_reader, args=(process.stdout, output_queue))
        reader_thread.daemon = True
        reader_thread.start()
        
        start_time = time.time()
        timeout = 160 # Increased timeout for M4 Pro loading + generation
        
        while True:
            # Check for new output lines
            try:
                line = output_queue.get_nowait()
                print(f"  [uv output] {line.strip()}", flush=True)
            except queue.Empty:
                # No output, check if process finished
                if process.poll() is not None:
                    break
                
                # Check for timeout
                if time.time() - start_time > timeout:
                    process.kill()
                    return f"Error: Command timed out after {timeout} seconds."
                
                time.sleep(0.1) # Avoid busy loop
        
        # Wait for thread to finish (it should have since pipe is closed)
        reader_thread.join(timeout=1)
        
        if process.returncode == 0:
            return f"Success: Audio saved to {output_path}"
        else:
            return f"Error: Command failed with exit code {process.returncode}"
            
    except Exception as e:
        if 'process' in locals(): process.kill()
        return f"Error: An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    # Simple CLI for the tool
    import argparse
    parser = argparse.ArgumentParser(description="Qwen3-TTS optimized tool for Antigravity")
    parser.add_argument("text", help="Text to convert")
    parser.add_argument("-i", "--instruction", help="Voice description")
    parser.add_argument("-c", "--clone", help="Path to voice sample for cloning")
    parser.add_argument("-o", "--output", default="output.wav", help="Output file path")
    
    args = parser.parse_args()
    print(generate_speech(args.text, args.instruction, args.clone, args.output))
