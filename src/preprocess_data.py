import torch
import numpy as np
from PIL import Image
import h5py
import pickle
from tqdm import tqdm

class DataPreprocessor:
    """Предварительная обработка и сохранение данных в быстром формате"""
    
    @staticmethod
    def preprocess_to_hdf5(config):
        """Сохранение данных в HDF5 формате"""
        
        h5_path = config.DATA_PATH / 'gtsrb_preprocessed.h5'
        
        with h5py.File(h5_path, 'w') as h5f:
            # Создание датасетов
            train_images = h5f.create_dataset(
                'train_images', 
                shape=(num_train, 3, 224, 224),
                dtype='float32'
            )
            train_labels = h5f.create_dataset(
                'train_labels', 
                shape=(num_train,),
                dtype='int64'
            )
            
            # Заполнение данными
            for idx, img_path in enumerate(tqdm(train_paths)):
                img = Image.open(img_path).convert('RGB')
                img = transform(img)
                train_images[idx] = img.numpy()
                train_labels[idx] = label
    
    @staticmethod
    def load_from_hdf5(config):
        """Быстрая загрузка из HDF5"""
        import h5py
        
        h5_path = config.DATA_PATH / 'gtsrb_preprocessed.h5'
        with h5py.File(h5_path, 'r') as h5f:
            train_images = torch.from_numpy(h5f['train_images'][:])
            train_labels = torch.from_numpy(h5f['train_labels'][:])
        
        return torch.utils.data.TensorDataset(train_images, train_labels)