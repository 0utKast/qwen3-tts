import torch
<<<<<<< HEAD
import platform

print(f"Python Version: {platform.python_version()}")
print(f"PyTorch Version: {torch.__version__}")
print(f"MPS Available: {torch.backends.mps.is_available()}")
print(f"MPS Built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")
    x = torch.ones(1, device=mps_device)
    print(f"Successfully created tensor on MPS: {x}")
else:
    print("MPS not available.")
=======

def check_mps():
    print(f"PyTorch Version: {torch.__version__}")
    
    if not torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is NOT available.")
        return
    
    print("MPS is available.")
    
    try:
        device = torch.device("mps")
        x = torch.ones(5, device=device)
        y = torch.zeros(5, device=device)
        z = x + y
        print(f"Tensor operation on MPS successful: {z}")
        
        # Check bfloat16 support (critical for Qwen3-TTS on M4)
        try:
            bf16_tensor = torch.zeros(5, dtype=torch.bfloat16, device=device)
            print("bfloat16 is supported on this MPS device.")
        except Exception as e:
            print(f"bfloat16 NOT supported: {e}")
            
    except Exception as e:
        print(f"An error occurred while using MPS: {e}")

if __name__ == "__main__":
    check_mps()
>>>>>>> 51bb89b (feat: v1.3.0 - Soporte Mac M1-M4, Streaming en tiempo real y diseño de voz consistente)
