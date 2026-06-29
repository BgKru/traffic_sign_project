import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
import torchvision.transforms.autoaugment as autoaugment
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


class RandomErasing:
    """Random Erasing аугментация для улучшения регуляризации"""
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.485, 0.456, 0.406)):
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.mean = mean
    
    def __call__(self, img):
        if torch.rand(1) > self.probability:
            return img
        
        h, w = img.shape[1], img.shape[2]
        area = h * w
        
        # Случайный выбор размера области
        target_area = np.random.uniform(self.sl, self.sh) * area
        aspect_ratio = np.random.uniform(self.r1, 1 / self.r1)
        
        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))
        
        if h_erase < h and w_erase < w:
            x = np.random.randint(0, w - w_erase)
            y = np.random.randint(0, h - h_erase)
            
            # Заполнение случайным цветом или средним
            if torch.rand(1) > 0.5:
                img[:, y:y+h_erase, x:x+w_erase] = torch.tensor(self.mean).view(3, 1, 1)
            else:
                img[:, y:y+h_erase, x:x+w_erase] = torch.rand(3, h_erase, w_erase)
        
        return img

class GTSRBDatasetMobileNet(Dataset):
    """Оптимизированный датасет для MobileNet"""
    
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
        
        self.class_counts = self.data['ClassId'].value_counts().to_dict()
        self.classes = sorted(self.data['ClassId'].unique())
        self.num_classes = len(self.classes)
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

def get_mobilenet_transforms(input_size=224, augment=True, use_auto_augment=True, use_random_erasing=False):
    """
    Трансформации для MobileNetV3 с усиленными аугментациями
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
        
        # Усиленные аугментации для MobileNet
        train_transform_list.extend([
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip(p=0.5),  # MobileNet выигрывает от горизонтальных отражений
            transforms.RandomGrayscale(p=0.05),  # Иногда переводим в черно-белое
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
        if use_random_erasing:
            train_transform_list.append(RandomErasing(probability=0.3))
        
        return transforms.Compose(train_transform_list)
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

def get_mobilenet_data_loaders(config, device=None):
    """
    Создает DataLoader'ы для MobileNet
    """
    train_transform = get_mobilenet_transforms(
        config.INPUT_SIZE, 
        augment=True, 
        use_auto_augment=getattr(config, 'AUTO_AUGMENT', True),
        use_random_erasing=getattr(config, 'RANDOM_ERASING', False)
    )
    val_transform = get_mobilenet_transforms(config.INPUT_SIZE, augment=False)
    test_transform = get_mobilenet_transforms(config.INPUT_SIZE, augment=False)
    
    train_dataset = GTSRBDatasetMobileNet(
        root_dir=config.DATA_PATH,
        transform=train_transform,
        split='train',
        val_size=0.2
    )
    
    val_dataset = GTSRBDatasetMobileNet(
        root_dir=config.DATA_PATH,
        transform=val_transform,
        split='val',
        val_size=0.2
    )
    
    test_dataset = GTSRBDatasetMobileNet(
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
    from configs.mobilenet_config import MobileNetV3Config
    from src.utils import GPUManager
    
    gpu_manager = GPUManager()
    config = MobileNetV3Config
    train_loader, val_loader, test_loader = get_mobilenet_data_loaders(config, gpu_manager.device)
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    images, labels = next(iter(train_loader))
    print(f"Image batch shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")