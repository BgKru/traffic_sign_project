import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, Optional, Dict
import gc

class GPUManager:
    """Управление GPU и оптимизация памяти"""
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Инициализация менеджера GPU
        
        Args:
            device: Устройство для использования (если None, автоматический выбор)
        """
        self.device = self.get_best_device(device)
        self.gpu_info = self.get_gpu_info()
        
    def get_best_device(self, preferred_device: Optional[torch.device] = None) -> torch.device:
        """
        Выбор лучшего доступного устройства
        
        Args:
            preferred_device: Предпочтительное устройство
        Returns:
            torch.device: Выбранное устройство
        """
        if preferred_device is not None:
            return preferred_device
        
        if torch.cuda.is_available():
            # Выбираем GPU с наибольшей свободной памятью
            if torch.cuda.device_count() > 1:
                max_free_memory = 0
                best_gpu = 0
                for i in range(torch.cuda.device_count()):
                    try:
                        torch.cuda.set_device(i)
                        free_memory = torch.cuda.memory_stats(i)['allocated_bytes.all.current'] / 1e9
                        if free_memory > max_free_memory:
                            max_free_memory = free_memory
                            best_gpu = i
                    except:
                        continue
                return torch.device(f'cuda:{best_gpu}')
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
    
    def get_gpu_info(self) -> Dict:
        """
        Получение информации о GPU
        
        Returns:
            Dict: Информация о GPU
        """
        info = {
            'device': self.device,
            'device_name': str(self.device),
            'is_cuda': self.device.type == 'cuda'
        }
        
        if self.device.type == 'cuda':
            info.update({
                'gpu_name': torch.cuda.get_device_name(self.device),
                'gpu_count': torch.cuda.device_count(),
                'total_memory_gb': torch.cuda.get_device_properties(self.device).total_memory / 1e9,
                'cuda_version': torch.version.cuda,
                'cudnn_version': torch.backends.cudnn.version()
            })
            
            # Текущее использование памяти
            try:
                allocated = torch.cuda.memory_allocated(self.device) / 1e9
                reserved = torch.cuda.memory_reserved(self.device) / 1e9
                info['allocated_memory_gb'] = allocated
                info['reserved_memory_gb'] = reserved
                info['free_memory_gb'] = info['total_memory_gb'] - allocated
            except:
                info['allocated_memory_gb'] = 0
                info['reserved_memory_gb'] = 0
                info['free_memory_gb'] = info['total_memory_gb']
        
        return info
    
    def print_gpu_info(self):
        """Вывод информации о GPU"""
        print("=" * 50)
        print("GPU INFORMATION")
        print("=" * 50)
        print(f"Device: {self.gpu_info['device']}")
        print(f"Is CUDA: {self.gpu_info['is_cuda']}")
        
        if self.gpu_info['is_cuda']:
            print(f"GPU Name: {self.gpu_info['gpu_name']}")
            print(f"GPU Count: {self.gpu_info['gpu_count']}")
            print(f"Total Memory: {self.gpu_info['total_memory_gb']:.2f} GB")
            print(f"Allocated Memory: {self.gpu_info.get('allocated_memory_gb', 0):.2f} GB")
            print(f"Free Memory: {self.gpu_info.get('free_memory_gb', 0):.2f} GB")
            print(f"CUDA Version: {self.gpu_info['cuda_version']}")
            print(f"CUDNN Version: {self.gpu_info['cudnn_version']}")
        print("=" * 50)
    
    @staticmethod
    def clear_gpu_memory():
        """Очистка памяти GPU"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
    
    @staticmethod
    def get_memory_usage(device: torch.device) -> Dict:
        """
        Получение текущего использования памяти
        
        Args:
            device: Устройство
        Returns:
            Dict: Информация об использовании памяти
        """
        if device.type != 'cuda':
            return {'device': 'cpu', 'memory_usage': 'N/A'}
        
        try:
            allocated = torch.cuda.memory_allocated(device) / 1e9
            reserved = torch.cuda.memory_reserved(device) / 1e9
            total = torch.cuda.get_device_properties(device).total_memory / 1e9
            
            return {
                'device': str(device),
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'total_gb': total,
                'free_gb': total - allocated,
                'utilization': (allocated / total) * 100
            }
        except:
            return {'device': str(device), 'memory_usage': 'Error'}
    
    @staticmethod
    def to_device(data, device: torch.device):
        """
        Перемещение данных на устройство с оптимизацией
        
        Args:
            data: Данные для перемещения
            device: Целевое устройство
        Returns:
            Данные на устройстве
        """
        if isinstance(data, (list, tuple)):
            return [GPUManager.to_device(item, device) for item in data]
        elif isinstance(data, dict):
            return {k: GPUManager.to_device(v, device) for k, v in data.items()}
        elif isinstance(data, torch.Tensor):
            return data.to(device, non_blocking=True) if device.type == 'cuda' else data.to(device)
        else:
            return data

def setup_gpu_for_training(config) -> Tuple[torch.device, GPUManager]:
    """
    Настройка GPU для обучения
    
    Args:
        config: Конфигурация с параметрами обучения
    Returns:
        device: Устройство для обучения
        gpu_manager: Менеджер GPU
    """
    gpu_manager = GPUManager()
    
    # Настройка для обучения
    if gpu_manager.device.type == 'cuda':
        # Установка оптимизаций CUDA
        torch.backends.cudnn.benchmark = True  # Автоматический поиск оптимальных алгоритмов
        torch.backends.cudnn.deterministic = False  # Разрешить недетерминированные алгоритмы для скорости
        torch.backends.cudnn.enabled = True
        
        # Настройка использования памяти
        if hasattr(config, 'MEMORY_FRACTION'):
            torch.cuda.set_per_process_memory_fraction(config.MEMORY_FRACTION)
        
        # Получение информации
        gpu_manager.print_gpu_info()
    
    return gpu_manager.device, gpu_manager

def profile_model_performance(model: nn.Module, device: torch.device, 
                              input_size: Tuple = (1, 3, 224, 224)) -> Dict:
    """
    Профилирование производительности модели на GPU
    
    Args:
        model: Модель для профилирования
        device: Устройство
        input_size: Размер входного тензора
    Returns:
        Dict: Статистика производительности
    """
    model.eval()
    model.to(device)
    
    # Создание тестового входа
    x = torch.randn(input_size, device=device)
    
    # Прогрев
    for _ in range(20):
        with torch.no_grad():
            _ = model(x)
    
    # Измерение времени
    times = []
    memory_usage = []
    
    for _ in range(100):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        with torch.no_grad():
            _ = model(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        end_time = time.time()
        times.append((end_time - start_time) * 1000)  # мс
        
        # Запись использования памяти (каждые 10 прогонов)
        if _ % 10 == 0 and device.type == 'cuda':
            try:
                mem = torch.cuda.memory_allocated(device) / 1e9
                memory_usage.append(mem)
            except:
                pass
    
    # Очистка
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Статистика
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    results = {
        'avg_inference_time_ms': avg_time,
        'std_inference_time_ms': std_time,
        'min_inference_time_ms': min_time,
        'max_inference_time_ms': max_time,
        'fps': 1000 / avg_time,
        'device': str(device),
        'input_size': input_size,
        'num_runs': len(times)
    }
    
    if memory_usage:
        results['avg_memory_gb'] = np.mean(memory_usage)
        results['max_memory_gb'] = np.max(memory_usage)
    
    return results

class GPUMemoryMonitor:
    """Мониторинг использования GPU памяти в реальном времени"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.history = []
        self.is_active = False
        
    def start(self):
        """Начать мониторинг"""
        self.is_active = True
        self.history = []
        
        if self.device.type == 'cuda':
            # Сброс статистики
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
    
    def step(self):
        """Запись текущего состояния памяти"""
        if self.is_active and self.device.type == 'cuda':
            try:
                stats = {
                    'timestamp': time.time(),
                    'allocated': torch.cuda.memory_allocated(self.device) / 1e9,
                    'reserved': torch.cuda.memory_reserved(self.device) / 1e9,
                    'peak_allocated': torch.cuda.max_memory_allocated(self.device) / 1e9
                }
                self.history.append(stats)
            except:
                pass
    
    def stop(self) -> Dict:
        """Остановить мониторинг и вернуть статистику"""
        self.is_active = False
        
        if not self.history:
            return {}
        
        # Расчет статистики
        allocated = [h['allocated'] for h in self.history]
        reserved = [h['reserved'] for h in self.history]
        peak = [h['peak_allocated'] for h in self.history]
        
        stats = {
            'avg_allocated_gb': np.mean(allocated),
            'max_allocated_gb': np.max(allocated),
            'min_allocated_gb': np.min(allocated),
            'avg_reserved_gb': np.mean(reserved),
            'max_reserved_gb': np.max(reserved),
            'peak_allocated_gb': max(peak),
            'num_measurements': len(self.history)
        }
        
        return stats
    
    def plot_history(self, save_path: Optional[str] = None):
        """Визуализация истории использования памяти"""
        if not self.history:
            print("No history to plot")
            return
        
        import matplotlib.pyplot as plt
        
        times = [h['timestamp'] - self.history[0]['timestamp'] for h in self.history]
        allocated = [h['allocated'] for h in self.history]
        reserved = [h['reserved'] for h in self.history]
        peak = [h['peak_allocated'] for h in self.history]
        
        plt.figure(figsize=(12, 6))
        plt.plot(times, allocated, label='Allocated', linewidth=2)
        plt.plot(times, reserved, label='Reserved', linewidth=2)
        plt.plot(times, peak, label='Peak Allocated', linewidth=2)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Memory (GB)')
        plt.title('GPU Memory Usage Over Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()