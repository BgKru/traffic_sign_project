import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import torch

class GTSRBDataset(Dataset):
    """Датасет для GTSRB"""
    
    def __init__(self, root_dir, transform=None, split='train', val_size=0.2, random_state=42):
        """
        Args:
            root_dir: Путь к папке с датасетом GTSRB
            transform: Трансформации для изображений
            split: 'train', 'val' или 'test'
            val_size: Размер валидационной выборки от train
            random_state: Seed для воспроизводимости
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # Загрузка данных
        if split == 'test':
            # Тестовые данные
            test_csv = os.path.join(root_dir, 'Test.csv')
            self.data = pd.read_csv(test_csv)
            self.data['Path'] = self.data['Path'].apply(
                lambda x: os.path.join(root_dir, x.replace('./', ''))
            )
        else:
            # Тренировочные данные
            train_csv = os.path.join(root_dir, 'Train.csv')
            full_data = pd.read_csv(train_csv)
            full_data['Path'] = full_data['Path'].apply(
                lambda x: os.path.join(root_dir, x.replace('./', ''))
            )
            
            # Разбиение на train/val
            train_data, val_data = train_test_split(
                full_data, 
                test_size=val_size, 
                random_state=random_state, 
                stratify=full_data['ClassId']
            )
            
            self.data = train_data if split == 'train' else val_data
            
        # Группировка по классам для балансировки
        self.class_counts = self.data['ClassId'].value_counts().to_dict()
        self.classes = sorted(self.data['ClassId'].unique())
        self.num_classes = len(self.classes)
        
        # Кэширование путей для ускорения
        self.image_paths = self.data['Path'].values
        self.labels = self.data['ClassId'].values
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_path = self.image_paths[idx]
        class_id = int(self.labels[idx])
        
        try:
            # Загрузка изображения
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Fallback: черное изображение
            image = Image.new('RGB', (224, 224), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        return image, class_id

def get_transforms(input_size=224, augment=True):
    """
    Возвращает трансформации для train/val/test
    
    Args:
        input_size: Размер входного изображения
        augment: Использовать аугментации для train
    """
    # Нормализация для ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augment:
        # Аугментации для обучения
        train_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return train_transform
    else:
        # Трансформации для валидации и теста
        val_transform = transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        return val_transform

def get_data_loaders(config):
    """
    Создает DataLoader'ы для train/val/test
    
    Args:
        config: Объект конфигурации
    Returns:
        train_loader, val_loader, test_loader
    """
    
    # Трансформации
    train_transform = get_transforms(config.INPUT_SIZE, augment=True)
    val_transform = get_transforms(config.INPUT_SIZE, augment=False)
    test_transform = get_transforms(config.INPUT_SIZE, augment=False)
    
    # Датасеты
    train_dataset = GTSRBDataset(
        root_dir=config.DATA_PATH,
        transform=train_transform,
        split='train',
        val_size=0.2
    )
    
    val_dataset = GTSRBDataset(
        root_dir=config.DATA_PATH,
        transform=val_transform,
        split='val',
        val_size=0.2
    )
    
    test_dataset = GTSRBDataset(
        root_dir=config.DATA_PATH,
        transform=test_transform,
        split='test'
    )
    
    # Опции DataLoader для производительности
    loader_kwargs = {
        'num_workers': 2,
        'pin_memory': True,
        'prefetch_factor': 2,
        'persistent_workers': True
    } if torch.cuda.is_available() else {
        'num_workers': 0
    }
    
    # DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        **loader_kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        **loader_kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        **loader_kwargs
    )
    
    return train_loader, val_loader, test_loader

# Пример использования для проверки загрузки данных
if __name__ == '__main__':
    from configs.efficientnet_config import EfficientNetConfig
    
    # Проверка загрузки
    train_loader, val_loader, test_loader = get_data_loaders(EfficientNetConfig)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Проверка одного батча
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique classes in batch: {torch.unique(labels)}")
    print(f"Number of classes: {len(train_loader.dataset.classes)}")
    
    # Проверка распределения классов
    print(f"\nClass distribution in train:")
    for class_id, count in sorted(train_loader.dataset.class_counts.items())[:5]:
        print(f"  Class {class_id}: {count} samples")