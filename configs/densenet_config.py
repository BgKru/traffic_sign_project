# configs/densenet_config.py

import os
from pathlib import Path

class DenseNetConfig:
    """Конфигурация для обучения DenseNet-121 с GPU оптимизациями"""
    
    # Пути к данным
    BASE_PATH = Path(__file__).parent.parent
    DATA_PATH = BASE_PATH / 'data' / 'GTSRB'
    TRAIN_PATH = DATA_PATH / 'train'
    TEST_PATH = DATA_PATH / 'test'
    
    # Параметры модели
    MODEL_NAME = 'densenet121_gpu'
    NUM_CLASSES = 43
    INPUT_SIZE = 224
    
    # Параметры обучения
    BATCH_SIZE = 32  # Уменьшен для стабильности на GPU
    EPOCHS = 15
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    
    # GPU оптимизации - отключаем ручное FP16, используем только AMP
    USE_HALF_PRECISION = False  # Отключаем ручное FP16
    MEMORY_FRACTION = 0.95
    
    # Аугментации
    USE_AUGMENTATION = True
    ROTATION_DEGREES = 15
    BRIGHTNESS = 0.2
    CONTRAST = 0.2
    HUE = 0.1
    AUTO_AUGMENT = True
    
    # Пути для сохранения
    RUNS_PATH = BASE_PATH / 'runs' / MODEL_NAME
    CHECKPOINTS_PATH = RUNS_PATH / 'checkpoints'
    LOGS_PATH = RUNS_PATH / 'logs'
    RESULTS_PATH = RUNS_PATH / 'results'
    
    # Устройство
    DEVICE = 'cuda'
    
    @classmethod
    def create_dirs(cls):
        """Создание директорий"""
        for path in [cls.RUNS_PATH, cls.CHECKPOINTS_PATH, cls.LOGS_PATH, cls.RESULTS_PATH]:
            path.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_class_names(cls):
        """Названия классов"""
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