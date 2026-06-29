import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms

class ResNet50Classifier(nn.Module):
    """Классификатор ResNet-50 с поддержкой GPU"""
    
    def __init__(self, num_classes=43, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        
        # Загрузка предобученной модели
        if pretrained:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Замена классификатора
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
        self.model_name = 'resnet50'
        self.input_size = 224
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_total_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self):
        return self.get_total_params() * 4 / (1024 * 1024)

def create_resnet50(config):
    """Создание модели с логированием параметров"""
    model = ResNet50Classifier(
        num_classes=config.NUM_CLASSES,
        pretrained=True
    )
    
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    return model