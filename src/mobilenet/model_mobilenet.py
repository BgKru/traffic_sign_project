# src/model_mobilenet.py

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights

class MobileNetV3LargeGPU(nn.Module):
    """
    Классификатор на основе MobileNetV3-Large
    """
    
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Количество классов
            pretrained: Использовать предобученные веса
            freeze_backbone: Заморозить backbone
        """
        super(MobileNetV3LargeGPU, self).__init__()
        
        # Загрузка предобученной модели
        if pretrained:
            self.backbone = models.mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
        else:
            self.backbone = models.mobilenet_v3_large(weights=None)
        
        # Получение размерности признаков
        num_features = self.backbone.classifier[0].in_features
        
        # Оптимизированный классификатор
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Заморозка backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        
        self.model_name = 'mobilenetv3_large'
        self.input_size = 224
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_feature_extractor(self):
        return nn.Sequential(*list(self.backbone.children())[:-1])
    
    def freeze_batch_norm(self):
        for module in self.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def unfreeze_all(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self):
        param_size = self.get_total_params()
        return param_size * 4 / (1024 * 1024)

def create_mobilenetv3_gpu(config):
    """Создает модель MobileNetV3-Large"""
    model = MobileNetV3LargeGPU(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False
    )
    
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    return model