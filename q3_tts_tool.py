import subprocess
import os
import sys

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

    cmd = [
        "uv", "run", 
        script_path, 
        text, 
        "-o", output_path
    ]
    
    if instruction:
        cmd.extend(["-i", instruction])
    
    if clone_path:
        if not os.path.exists(clone_path):
            return f"Error: Clone path '{clone_path}' does not exist."
        cmd.extend(["--clone", clone_path])
    
    print(f"Executing: {' '.join(cmd)}", flush=True)
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
        for line in process.stdout:
            print(f"  [uv output] {line.strip()}", flush=True)
        
        # Wait for completion with timeout
        process.wait(timeout=90)
        
        if process.returncode == 0:
            return f"Success: Audio saved to {output_path}"
        else:
            return f"Error: Command failed with exit code {process.returncode}"
            
    except subprocess.TimeoutExpired:
        process.kill()
        return f"Error: Command timed out after 90 seconds."
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
