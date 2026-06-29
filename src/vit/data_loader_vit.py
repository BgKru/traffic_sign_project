import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.autoaugment as autoaugment
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class MixupCutMixDataset:
    """
    Обертка для датасета с поддержкой Mixup и CutMix аугментаций
    
    Mixup: смешивание изображений и меток
    CutMix: замена области одного изображения на другое
    Эти аугментации критически важны для ViT
    """
    
    def __init__(self, dataset, mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=43):
        self.dataset = dataset
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.num_classes = num_classes
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # С вероятностью 0.5 применяем Mixup или CutMix
        if random.random() < 0.5:
            # Случайный выбор второго изображения
            idx2 = random.randint(0, len(self.dataset) - 1)
            img2, label2 = self.dataset[idx2]
            
            if random.random() < 0.5:
                # Mixup
                lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
                img = lam * img + (1 - lam) * img2
                label = (lam * torch.nn.functional.one_hot(
                    torch.tensor(label), num_classes=self.num_classes
                ).float() + 
                (1 - lam) * torch.nn.functional.one_hot(
                    torch.tensor(label2), num_classes=self.num_classes
                ).float())
            else:
                # CutMix
                lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                
                # Случайная область для замены
                h, w = img.shape[1], img.shape[2]
                cx = random.randint(0, w)
                cy = random.randint(0, h)
                
                rw = int(w * np.sqrt(1 - lam))
                rh = int(h * np.sqrt(1 - lam))
                
                x1 = max(0, cx - rw // 2)
                x2 = min(w, cx + rw // 2)
                y1 = max(0, cy - rh // 2)
                y2 = min(h, cy + rh // 2)
                
                img[:, y1:y2, x1:x2] = img2[:, y1:y2, x1:x2]
                
                label = (lam * torch.nn.functional.one_hot(
                    torch.tensor(label), num_classes=self.num_classes
                ).float() + 
                (1 - lam) * torch.nn.functional.one_hot(
                    torch.tensor(label2), num_classes=self.num_classes
                ).float())
        else:
            label = torch.nn.functional.one_hot(
                torch.tensor(label), num_classes=self.num_classes
            ).float()
        
        return img, label

class GTSRBDatasetViT(Dataset):
    """Датасет для ViT"""
    
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

def get_vit_transforms(input_size=224, augment=True, use_auto_augment=True):
    """
    Трансформации для ViT с усиленными аугментациями
    
    ViT требует более сильных аугментаций для хорошей обобщаемости
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if augment:
        train_transform_list = [
            transforms.Resize((input_size, input_size)),
        ]
        
        if use_auto_augment:
            # RandAugment или AutoAugment для ViT
            train_transform_list.append(
                autoaugment.AutoAugment(policy=autoaugment.AutoAugmentPolicy.IMAGENET)
            )
        
        train_transform_list.extend([
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),  # Вертикальное отражение полезно для дорожных знаков
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))  # Random Erasing для регуляризации
        ])
        
        return transforms.Compose(train_transform_list)
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_vit_data_loaders(config, device=None):
    """
    Создает DataLoader'ы для ViT с Mixup/CutMix
    """
    train_transform = get_vit_transforms(
        config.INPUT_SIZE, 
        augment=True, 
        use_auto_augment=getattr(config, 'AUTO_AUGMENT', True)
    )
    val_transform = get_vit_transforms(config.INPUT_SIZE, augment=False)
    test_transform = get_vit_transforms(config.INPUT_SIZE, augment=False)
    
    # Создание датасетов
    train_dataset = GTSRBDatasetViT(
        root_dir=config.DATA_PATH,
        transform=train_transform,
        split='train',
        val_size=0.2
    )
    
    # Обертывание для Mixup/CutMix
    train_dataset = MixupCutMixDataset(
        train_dataset,
        mixup_alpha=getattr(config, 'MIXUP_ALPHA', 0.8),
        cutmix_alpha=getattr(config, 'CUTMIX_ALPHA', 1.0),
        num_classes=config.NUM_CLASSES
    )
    
    val_dataset = GTSRBDatasetViT(
        root_dir=config.DATA_PATH,
        transform=val_transform,
        split='val',
        val_size=0.2
    )
    
    test_dataset = GTSRBDatasetViT(
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
    from configs.vit_config import ViTConfig
    from src.utils import GPUManager
    
    gpu_manager = GPUManager()
    config = ViTConfig
    train_loader, val_loader, test_loader = get_vit_data_loaders(config, gpu_manager.device)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")  # One-hot encoding для Mixup/CutMix