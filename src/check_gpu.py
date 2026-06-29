# src/diagnose.py

import sys
import platform
import subprocess
import torch
import os

def diagnose_system():
    """Полная диагностика системы для определения GPU"""
    
    print("=" * 70)
    print("SYSTEM DIAGNOSTIC")
    print("=" * 70)
    
    # Системная информация
    print(f"\n[1] System Information:")
    print(f"    OS: {platform.system()} {platform.release()}")
    print(f"    Architecture: {platform.machine()}")
    print(f"    Python: {sys.version}")
    print(f"    PyTorch: {torch.__version__}")
    
    # Проверка драйверов NVIDIA
    print(f"\n[2] NVIDIA Drivers:")
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("    NVIDIA drivers detected:")
            print("    " + result.stdout.split('\n')[1] if len(result.stdout.split('\n')) > 1 else "")
        else:
            print("    NVIDIA drivers NOT found")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("    NVIDIA drivers NOT found")
    
    # CUDA информация
    print(f"\n[3] CUDA Information:")
    print(f"    CUDA available: {torch.cuda.is_available()}")
    print(f"    CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}")
    print(f"    PyTorch CUDA build: {torch.version.cuda}")
    
    if torch.cuda.is_available():
        print(f"\n[4] GPU Information:")
        print(f"    Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    else:
        print("\n[4] GPU NOT available. Possible reasons:")
        print("    - No NVIDIA GPU installed")
        print("    - NVIDIA drivers not installed")
        print("    - CUDA Toolkit not installed")
        print("    - PyTorch installed without CUDA support")
        
        # Проверка, какая версия PyTorch установлена
        print(f"\n[5] PyTorch Installation:")
        print(f"    PyTorch version: {torch.__version__}")
        print(f"    Is built with CUDA: {torch.cuda.is_built()}")
        
        if not torch.cuda.is_built():
            print("\n    ⚠️ PyTorch was installed WITHOUT CUDA support!")
            print("    You have two options:")
            print("    1. Install CUDA version: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
            print("    2. Continue with CPU training (slow)")
    
    print("\n" + "=" * 70)
    return torch.cuda.is_available()

if __name__ == '__main__':
    diagnose_system()