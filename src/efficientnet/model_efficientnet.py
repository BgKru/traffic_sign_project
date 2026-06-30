# src/model_efficientnet.py

import torch
import torch.nn as nn
from torchvision import models
import timm

class EfficientNetB0Classifier(nn.Module):
    """
    Классификатор на основе EfficientNet-B0
    """
    
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Количество классов для классификации
            pretrained: Использовать предобученные веса
            freeze_backbone: Заморозить веса backbone
        """
        super(EfficientNetB0Classifier, self).__init__()
        
        # Загрузка предобученной модели через timm
        if pretrained:
            self.backbone = timm.create_model('efficientnet_b0', pretrained=True)
        else:
            self.backbone = timm.create_model('efficientnet_b0', pretrained=False)
        
        # Получение размерности признаков перед классификатором
        num_features = self.backbone.classifier.in_features
        
        # Замена классификатора
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
        # Заморозка backbone при необходимости
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        
        self.model_name = 'efficientnet_b0'
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

def create_efficientnet_b0(config):
    """Создает и настраивает модель EfficientNet-B0"""
    model = EfficientNetB0Classifier(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False
    )
    
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    return model