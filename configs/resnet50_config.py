import os
from pathlib import Path
import torch

class ResNet50Config:
    """Конфигурация для обучения ResNet-50 с поддержкой GPU"""
    
    # Пути к данным
    BASE_PATH = Path(__file__).parent.parent
    DATA_PATH = BASE_PATH / 'data' / 'GTSRB'
    TRAIN_PATH = DATA_PATH / 'train'
    TEST_PATH = DATA_PATH / 'test'
    
    # Параметры модели
    MODEL_NAME = 'resnet50'
    NUM_CLASSES = 43
    INPUT_SIZE = 224
    
    # Параметры обучения (оптимизированы для GPU)
    BATCH_SIZE = 128  # Хорошо для GPU с 8-12GB VRAM
    EPOCHS = 10
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # GPU оптимизации
    USE_AMP = True  # Automatic Mixed Precision
    NUM_WORKERS = 4  # Количество потоков для загрузки данных
    
    # Аугментации
    USE_AUGMENTATION = True
    ROTATION_DEGREES = 15
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    HUE = 0.1
    
    # Пути для сохранения
    RUNS_PATH = BASE_PATH / 'runs' / MODEL_NAME
    CHECKPOINTS_PATH = RUNS_PATH / 'checkpoints'
    LOGS_PATH = RUNS_PATH / 'logs'
    RESULTS_PATH = RUNS_PATH / 'results'
    
    @classmethod
    def create_dirs(cls):
        """Создание необходимых директорий"""
        for path in [cls.RUNS_PATH, cls.CHECKPOINTS_PATH, cls.LOGS_PATH, cls.RESULTS_PATH]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_device(cls):
        """Автоматическое определение устройства"""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            return device
        else:
            print("CUDA not available, using CPU")
            return torch.device('cpu')
    
    @classmethod
    def get_class_names(cls):
        """Возвращает названия классов (для GTSRB)"""
        return [
            'Speed limit 20', 'Speed limit 30', 'Speed limit 50', 
            'Speed limit 60', 'Speed limit 70', 'Speed limit 80', 
            'End of speed limit 80', 'Speed limit 100', 'Speed limit 120',
            'No overtaking', 'No overtaking (trucks)', 'Priority at intersection',
            'Priority road', 'Yield', 'Stop', 'No vehicles', 'No trucks',
            'No entry', 'General caution', 'Dangerous curve left',
            'Dangerous curve right', 'Double curve', 'Bumpy road',
            'Slippery road', 'Road narrows', 'Road work', 'Traffic signals',
            'Pedestrians', 'Children crossing', 'Bicycles crossing',
            'Beware of ice/snow', 'Wild animals crossing', 'End of all speed limits',
            'Turn right ahead', 'Turn left ahead', 'Ahead only',
            'Go straight or right', 'Go straight or left', 'Keep right',
            'Keep left', 'Roundabout', 'End of no overtaking',
            'End of no overtaking (trucks)'
        ]