import torch
import gc

def clear_cuda_memory():
    """Очистка памяти CUDA"""
    if torch.cuda.is_available():
        # Очистка кэша аллокатора PyTorch
        torch.cuda.empty_cache()
        
        # Синхронизация всех потоков CUDA
        torch.cuda.synchronize()
        
        # Принудительный сбор мусора Python
        gc.collect()
        
        # Дополнительная очистка для всех устройств
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

# Использование
clear_cuda_memory()