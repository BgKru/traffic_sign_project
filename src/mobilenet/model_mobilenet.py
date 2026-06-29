# src/model_mobilenet.py

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights

class MobileNetV3LargeGPU(nn.Module):
    """
    Оптимизированная для GPU модель MobileNetV3-Large
    
    Особенности:
    - Глубинные разделимые свертки (Depthwise Separable Convolutions)
    - Squeeze-and-Excitation блоки
    - h-swish активация
    - Оптимизация для GPU
    """
    
    def __init__(self, num_classes=43, pretrained=True, freeze_backbone=False, use_half=False):
        """
        Args:
            num_classes: Количество классов
            pretrained: Использовать предобученные веса
            freeze_backbone: Заморозить backbone
            use_half: Использовать FP16
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
        
        # Оптимизированный классификатор для MobileNetV3
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.BatchNorm1d(1024),  # Стабилизация на GPU
            nn.Hardswish(),  # Использование h-swish (отличительная черта MobileNetV3)
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
        self.use_half = use_half
        
        # Оптимизация для GPU
        self._optimize_for_gpu()
    
    def _optimize_for_gpu(self):
        """Оптимизация модели для GPU"""
        if torch.cuda.is_available():
            # Использование channels_last формата для скорости
            self.backbone = self.backbone.to(memory_format=torch.channels_last)
            
            if self.use_half:
                self.backbone = self.backbone.half()
    
    def forward(self, x):
        """Прямой проход"""
        if self.use_half and x.dtype == torch.float32:
            x = x.half()
        return self.backbone(x)
    
    def get_feature_extractor(self):
        """Извлечение признаков без классификатора"""
        return nn.Sequential(*list(self.backbone.children())[:-1])
    
    def freeze_batch_norm(self):
        """Замораживание BatchNorm"""
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
    
    def get_flops(self, input_size=(1, 3, 224, 224)):
        """Приблизительный подсчет FLOPs"""
        try:
            from thop import profile
            x = torch.randn(input_size)
            if torch.cuda.is_available():
                x = x.cuda()
                self.backbone = self.backbone.cuda()
            flops, params = profile(self.backbone, inputs=(x,), verbose=False)
            return flops
        except:
            return 0

def create_mobilenetv3_gpu(config, device):
    """
    Создание модели MobileNetV3-Large с оптимизацией для GPU
    """
    use_half = getattr(config, 'USE_HALF_PRECISION', False)
    
    model = MobileNetV3LargeGPU(
        num_classes=config.NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False,
        use_half=use_half
    )
    
    model = model.to(device)
    
    # Информация о модели
    print(f"Model: {model.model_name}")
    print(f"Total parameters: {model.get_total_params():,}")
    print(f"Trainable parameters: {model.get_trainable_params():,}")
    print(f"Model size: {model.get_model_size_mb():.2f} MB")
    print(f"Device: {device}")
    print(f"Half precision: {use_half}")
    
    # Подсчет FLOPs
    flops = model.get_flops()
    if flops:
        print(f"FLOPs: {flops / 1e9:.2f}G")
    
    return model

# Проверка модели
if __name__ == '__main__':
    from configs.mobilenet_config import MobileNetV3Config
    from src.utils import GPUManager
    
    gpu_manager = GPUManager()
    device = gpu_manager.device
    
    model = create_mobilenetv3_gpu(MobileNetV3Config, device)
    
    # Тестовый forward
    x = torch.randn(1, 3, 224, 224).to(device)
    output = model(x)
    print(f"Output shape: {output.shape}")