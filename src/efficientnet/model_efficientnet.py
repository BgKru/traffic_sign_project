import torch
import torch.nn as nn
from torchvision import models
import timm

class EfficientNetB0Classifier(nn.Module):
    """
    Классификатор на основе EfficientNet-B0
    
    Архитектура:
    - Использует предобученные на ImageNet веса
    - Заменяет классификатор на 43 класса GTSRB
    - Использует MBConv блоки с SE (Squeeze-and-Excitation)
    """
    
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False):
        """
        Args:
            num_classes: Количество классов для классификации
            pretrained: Использовать предобученные веса
            freeze_backbone: Заморозить веса backbone (для feature extraction)
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
            # Размораживаем только последний слой
            for param in self.backbone.classifier.parameters():
                param.requires_grad = True
        
        # Сохраняем имя архитектуры для отчетов
        self.model_name = 'efficientnet_b0'
        self.input_size = 224
        
    def forward(self, x):
        """Прямой проход через сеть"""
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Возвращает модель без классификатора для извлечения признаков"""
        return nn.Sequential(*list(self.backbone.children())[:-1])
    
    def freeze_batch_norm(self):
        """Замораживание BatchNorm слоев (полезно при мелком batch size)"""
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

def create_efficientnet_b0(config, device=None):
    """
    Создает и настраивает модель EfficientNet-B0
    
    Args:
        config: Объект конфигурации
        device: Устройство для модели (если None, модель не перемещается)
    Returns:
        model: EfficientNetB0Classifier
    """
    model = EfficientNetB0Classifier(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False  # Будем дообучать все слои
    )
    
    # Перемещение на устройство, если указано
    if device is not None:
        model = model.to(device)
    
    # Логирование информации о модели
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    
    return model

# Альтернативный вариант с использованием torchvision
def create_efficientnet_b0_torchvision(config, device=None):
    """
    Создает модель EfficientNet-B0 через torchvision
    """
    from torchvision import models
    
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    # Замена классификатора
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.2),
        nn.Linear(512, config.NUM_CLASSES)
    )
    
    model.model_name = 'efficientnet_b0'
    model.input_size = 224
    
    # Перемещение на устройство, если указано
    if device is not None:
        model = model.to(device)
    
    # Логирование
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: efficientnet_b0")
    print(f"Total parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / (1024 * 1024):.2f} MB")
    
    return model

# Проверка модели
if __name__ == '__main__':
    from configs.efficientnet_config import EfficientNetConfig
    
    # Создание модели
    model = create_efficientnet_b0(EfficientNetConfig)
    
    # Тестовый forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # Подсчет FLOPs (приблизительно)
    try:
        from thop import profile
        flops, params = profile(model, inputs=(x,))
        print(f"FLOPs: {flops / 1e9:.2f}G")
        print(f"Parameters: {params / 1e6:.2f}M")
    except:
        print("Install thop for FLOPs measurement: pip install thop")