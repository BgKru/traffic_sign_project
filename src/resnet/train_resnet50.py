import os
import sys
import json
import time
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.resnet50_config import ResNet50Config
from src.data_loader import get_data_loaders
from src.resnet.model_resnet50 import create_resnet50

def convert_to_serializable(obj):
    """
    Рекурсивно преобразует объекты в JSON-сериализуемый формат
    """
    if isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class GPUTrainer:
    """GPU-оптимизированный тренер с поддержкой AMP"""
    
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Критерий и оптимизатор
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        
        # Планировщик
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=1e-6
        )
        
        # Mixed Precision Training
        self.use_amp = config.USE_AMP
        self.scaler = GradScaler() if self.use_amp else None
        
        # История
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_time': [],
            'learning_rates': []
        }
        
        config.create_dirs()
        self._print_gpu_info()
    
    def _print_gpu_info(self):
        """Вывод информации о GPU"""
        if self.device.type == 'cuda':
            print(f"\nGPU Information:")
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"Memory Allocated: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            print(f"Memory Cached: {torch.cuda.memory_reserved(0) / 1e9:.2f} GB")
            print(f"Mixed Precision: {self.use_amp}")
    
    def train_epoch(self, train_loader):
        """Обучение одной эпохи с AMP"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc='Training')):
            # Перемещение данных на GPU
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass с AMP
            if self.use_amp:
                with autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                # Backward pass с масштабированием
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Очистка кэша GPU
            if batch_idx % 50 == 0 and self.device.type == 'cuda':
                torch.cuda.empty_cache()
        
        epoch_time = time.time() - start_time
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        return train_loss, train_acc, epoch_time
    
    def validate(self, val_loader):
        """Валидация модели"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        return val_loss, val_acc
    
    def train(self, train_loader, val_loader, epochs=None):
        """Полный цикл обучения"""
        if epochs is None:
            epochs = self.config.EPOCHS
        
        print(f"\nStarting training on {self.device}")
        print(f"Model: {self.config.MODEL_NAME}")
        print(f"Total epochs: {epochs}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print("=" * 60)
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch [{epoch}/{epochs}]")
            print("-" * 40)
            
            # Обучение
            train_loss, train_acc, epoch_time = self.train_epoch(train_loader)
            
            # Валидация
            val_loss, val_acc = self.validate(val_loader)
            
            # Обновление планировщика
            self.scheduler.step()
            
            # Сохранение истории
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_time'].append(epoch_time)
            self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            # Логирование
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            # Информация о GPU
            if self.device.type == 'cuda':
                print(f"GPU Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"*** New best model! Val Acc: {val_acc:.2f}% ***")
            
            # Регулярное сохранение
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_acc)
        
        print("\n" + "=" * 60)
        print(f"Training completed! Best Val Acc: {best_val_acc:.2f}%")
        
        return self.history
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.config.CHECKPOINTS_PATH / 'best_model.pth'
        else:
            path = self.config.CHECKPOINTS_PATH / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Сохранение истории обучения"""
        history_path = self.config.LOGS_PATH / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        
        self.plot_history()
    
    def plot_history(self):
        """Построение графиков обучения"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # График потерь
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График точности
        ax2.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # График learning rate
        ax3.plot(self.history['learning_rates'], label='Learning Rate', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # График времени
        ax4.plot(self.history['train_time'], label='Epoch Time', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Training Time per Epoch')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_PATH / 'training_history.png', dpi=150)
        plt.show()

def evaluate_model_gpu(model, test_loader, config, device):
    """Оценка модели на GPU"""
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if config.USE_AMP:
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    cm = confusion_matrix(all_labels, all_preds)
    
    results = {
        'test_loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist()
    }
    
    # Сохранение результатов
    results_path = config.RESULTS_PATH / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Матрица ошибок
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(config.RESULTS_PATH / 'confusion_matrix.png', dpi=150)
    plt.show()
    
    return results, all_preds, all_labels

def generate_report_gpu(config, model, results, history):
    """Генерация отчета"""
    report = {
        'model_name': config.MODEL_NAME,
        'input_size': config.INPUT_SIZE,
        'num_classes': config.NUM_CLASSES,
        'batch_size': config.BATCH_SIZE,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'use_amp': config.USE_AMP,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'total_params': model.get_total_params(),
        'trainable_params': model.get_trainable_params(),
        'best_val_acc': max(history['val_acc']),
        'test_accuracy': results['accuracy'],
        'test_precision': results['precision'],
        'test_recall': results['recall'],
        'test_f1': results['f1_score'],
        'training_time': sum(history['train_time'])
    }
    
    report_path = config.RESULTS_PATH / 'model_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\n" + "=" * 60)
    print("SUMMARY REPORT")
    print("=" * 60)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

def main():
    """Основная функция"""
    
    # Конфигурация
    config = ResNet50Config
    
    # Определение устройства
    device = config.get_device()
    
    # Загрузка данных
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(f"Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
    print(f"Val: {len(val_loader.dataset)} samples, {len(val_loader)} batches")
    print(f"Test: {len(test_loader.dataset)} samples, {len(test_loader)} batches")
    
    # Создание модели
    print("\nCreating model...")
    model = create_resnet50(config)
    
    # Обучение
    print("\nStarting training...")
    trainer = GPUTrainer(model, config, device)
    history = trainer.train(train_loader, val_loader)
    
    # Сохранение истории
    trainer.save_history()
    
    # Оценка
    print("\nEvaluating on test set...")
    results, preds, labels = evaluate_model_gpu(model, test_loader, config, device)
    
    print("\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    # Сохранение модели
    final_model_path = config.RUNS_PATH / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Отчет
    generate_report_gpu(config, model, results, history)

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()