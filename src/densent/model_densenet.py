# src/model_densenet.py

import torch
import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights

class DenseNet121GPU(nn.Module):
    """
    Оптимизированная для GPU модель DenseNet-121
    
    Особенности:
    - Плотные связи (Dense Connections) между всеми слоями
    - Блоки с узкими слоями (growth rate = 32)
    - Переходные слои для уменьшения размерности
    - Эффективное переиспользование признаков
    """
    
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Количество классов
            pretrained: Использовать предобученные веса
            freeze_backbone: Заморозить backbone
        """
        super(DenseNet121GPU, self).__init__()
        
        # Загрузка предобученной модели
        if pretrained:
            self.backbone = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        else:
            self.backbone = densenet121(weights=None)
        
        # Получение размерности признаков
        num_features = self.backbone.classifier.in_features
        
        # Замена классификатора с BatchNorm для стабильности
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Заморозка backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        
        self.model_name = 'densenet121'
        self.input_size = 224
        
        # Оптимизация для GPU
        self._optimize_for_gpu()
    
    def _optimize_for_gpu(self):
        """Оптимизация модели для GPU"""
        if torch.cuda.is_available():
            try:
                self.backbone = self.backbone.to(memory_format=torch.channels_last)
            except:
                pass
    
    def forward(self, x):
        """Прямой проход"""
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Извлечение признаков без классификатора"""
        return nn.Sequential(*list(self.backbone.children())[:-1])
    
    def freeze_batch_norm(self):
        """Замораживание BatchNorm слоев"""
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        """Размораживание всех слоев"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        """Количество обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Общее количество параметров"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self):
        """Размер модели в МБ"""
        param_size = self.get_total_params()
        return param_size * 4 / (1024 * 1024)
    
    def get_feature_map_size(self):
        """
        Возвращает размеры карт признаков на каждом уровне
        Для анализа плотных связей
        """
        feature_info = {
            'block1': {'layers': 6, 'output_size': 56},
            'block2': {'layers': 12, 'output_size': 28},
            'block3': {'layers': 24, 'output_size': 14},
            'block4': {'layers': 16, 'output_size': 7}
        }
        return feature_info

def create_densenet121_gpu(config, device):
    """
    Создание модели DenseNet-121 с оптимизацией для GPU
    """
    model = DenseNet121GPU(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False
    )
    
    model = model.to(device)
    
    # Информация о модели
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Device: {device}")
    
    # Информация о плотных связях
    feature_info = model.get_feature_map_size()
    print("\nDenseNet-121 Architecture:")
    for block_name, info in feature_info.items():
        print(f"  {block_name}: {info['layers']} dense layers, output size {info['output_size']}x{info['output_size']}")
    
    return model

# Проверка модели
if __name__ == '__main__':
    from configs.densenet_config import DenseNetConfig
    from src.utils import GPUManager
    
    gpu_manager = GPUManager()
    device = gpu_manager.device
    
    model = create_densenet121_gpu(DenseNetConfig, device)
    
    # Тестовый forward
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")