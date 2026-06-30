# src/model_resnet50.py

import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

class ResNet50Classifier(nn.Module):
    """
    Классификатор на основе ResNet-50 с дообучением
    """
    
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Количество классов для классификации
            pretrained: Использовать предобученные веса
            freeze_backbone: Заморозить веса backbone
        """
        super(ResNet50Classifier, self).__init__()
        
        # Загрузка предобученной модели
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Получение размерности признаков перед классификатором
        num_features = self.backbone.fc.in_features
        
        # Замена классификатора
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        # Заморозка backbone при необходимости
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            # Размораживаем только последний слой
            for param in self.backbone.fc.parameters():
                param.requires_grad = True
        
        # Сохраняем имя архитектуры для отчетов
        self.model_name = 'resnet50'
        self.input_size = 224
        
    def forward(self, x):
        """Прямой проход через сеть"""
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Возвращает модель без классификатора для извлечения признаков"""
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
        """Возвращает количество обучаемых параметров"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_total_params(self):
        """Возвращает общее количество параметров"""
        return sum(p.numel() for p in self.parameters())
    
    def get_model_size_mb(self):
        """Возвращает размер модели в МБ"""
        param_size = self.get_total_params()
        return param_size * 4 / (1024 * 1024)  # float32 = 4 bytes

def create_resnet50(config):
    """Создает и настраивает модель ResNet-50"""
    model = ResNet50Classifier(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False
    )
    
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    return model