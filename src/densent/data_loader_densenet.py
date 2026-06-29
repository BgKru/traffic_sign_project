# src/data_loader_densenet.py

import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.autoaugment as autoaugment
from sklearn.model_selection import train_test_split
import torch

class GTSRBDatasetDenseNet(Dataset):
    """Датасет для DenseNet-121"""
    
    def __init__(self, root_dir, transform=None, split='train', val_size=0.2, random_state=42):
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        if split == 'test':
            test_csv = os.path.join(root_dir, 'Test.csv')
            self.data = pd.read_csv(test_csv)
            self.data['Path'] = self.data['Path'].apply(
                lambda x: os.path.join(root_dir, x.replace('./', ''))
            )
        else:
            train_csv = os.path.join(root_dir, 'Train.csv')
            full_data = pd.read_csv(train_csv)
            full_data['Path'] = full_data['Path'].apply(
                lambda x: os.path.join(root_dir, x.replace('./', ''))
            )
            
            train_data, val_data = train_test_split(
                full_data, 
                test_size=val_size, 
                random_state=random_state, 
                stratify=full_data['ClassId']
            )
            
            self.data = train_data if split == 'train' else val_data
        
        self.image_paths = self.data['Path'].values
        self.labels = self.data['ClassId'].values
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
        
        class_id = int(self.labels[idx])
        
        if self.transform:
            image = self.transform(image)
        
        return image, class_id

def get_densenet_transforms(input_size=224, augment=True, use_auto_augment=True):
    """
    Трансформации для DenseNet-121
    
    DenseNet хорошо работает с умеренными аугментациями
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augment:
        train_transform_list = [
            transforms.Resize((input_size, input_size)),
        ]
        
        if use_auto_augment:
            train_transform_list.append(
                autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET)
            )
        
        train_transform_list.extend([
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        return transforms.Compose(train_transform_list)
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_densenet_data_loaders(config, device=None):
    """
    Создает DataLoader'ы для DenseNet-121
    """
    train_transform = get_densenet_transforms(
        config.INPUT_SIZE, 
        augment=True, 
        use_auto_augment=getattr(config, 'AUTO_AUGMENT', True)
    )
    val_transform = get_densenet_transforms(config.INPUT_SIZE, augment=False)
    test_transform = get_densenet_transforms(config.INPUT_SIZE, augment=False)
    
    train_dataset = GTSRBDatasetDenseNet(
        root_dir=config.DATA_PATH,
        transform=train_transform,
        split='train',
        val_size=0.2
    )
    
    val_dataset = GTSRBDatasetDenseNet(
        root_dir=config.DATA_PATH,
        transform=val_transform,
        split='val',
        val_size=0.2
    )
    
    test_dataset = GTSRBDatasetDenseNet(
        root_dir=config.DATA_PATH,
        transform=test_transform,
        split='test'
    )
    
    loader_kwargs = {
        'num_workers': 4,
        'pin_memory': True if device and device.type == 'cuda' else False,
        'prefetch_factor': 2,
        'persistent_workers': True
    }
    
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

if __name__ == '__main__':
    from configs.densenet_config import DenseNetConfig
    from src.utils import GPUManager
    
    gpu_manager = GPUManager()
    config = DenseNetConfig
    train_loader, val_loader, test_loader = get_densenet_data_loaders(config, gpu_manager.device)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")