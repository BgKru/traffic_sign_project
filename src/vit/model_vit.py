# src/model_vit.py

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights

class VisionTransformerGPU(nn.Module):
    """
    Оптимизированная для GPU модель Vision Transformer
    """
    
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Количество классов
            pretrained: Использовать предобученные веса
            freeze_backbone: Заморозить backbone
        """
        super(VisionTransformerGPU, self).__init__()
        
        # Загрузка предобученной модели
        if pretrained:
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            self.backbone = vit_b_16(weights=None)
        
        # Получение размерности признаков
        num_features = self.backbone.heads.head.in_features
        
        # Замена классификатора
        self.backbone.heads.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Заморозка backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.heads.parameters():
                param.requires_grad = True
        
        self.model_name = 'vit_base_patch16_224'
        self.input_size = 224
        self._device = None
        
        # Оптимизация для GPU
        self._optimize_for_gpu()
    
    def _optimize_for_gpu(self):
        """Оптимизация модели для GPU"""
        if torch.cuda.is_available():
            try:
                self.backbone = self.backbone.to(memory_format=torch.contiguous_format)
            except:
                pass
    
    def to(self, device):
        """Перемещение модели на устройство"""
        self._device = device
        self.backbone = self.backbone.to(device)
        return self
    
    def forward(self, x):
        """Прямой проход"""
        # Убеждаемся, что вход на правильном устройстве
        if self._device is not None and x.device != self._device:
            x = x.to(self._device)
        
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Извлечение признаков без классификатора"""
        return nn.Sequential(*list(self.backbone.children())[:-1])
    
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

def create_vit_gpu(config, device):
    """
    Создание модели ViT с оптимизацией для GPU
    """
    model = VisionTransformerGPU(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False
    )
    
    model = model.to(device)
    
    # Информация о модели
    print(f"Model: {model.model_name}")
    print(f"Patch size: 16x16")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Device: {device}")
    
    return model

# Проверка модели
if __name__ == '__main__':
    from configs.vit_config import ViTConfig
    from src.utils import GPUManager
    
    gpu_manager = GPUManager()
    device = gpu_manager.device
    
    model = create_vit_gpu(ViTConfig, device)
    
    # Тестовый forward
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")