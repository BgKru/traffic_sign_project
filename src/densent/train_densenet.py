# src/train_densenet.py

import os
import sys
import json
import time
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler
torch.multiprocessing.set_sharing_strategy('file_system')
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from configs.densenet_config import DenseNetConfig
from src.densent.data_loader_densenet import get_densenet_data_loaders
from src.densent.model_densenet import create_densenet121_gpu
from src.utils import GPUManager, GPUMemoryMonitor, profile_model_performance

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

class DenseNetTrainer:
    """
    Тренер для DenseNet-121 с GPU оптимизациями
    
    Особенности:
    - Простая и стабильная архитектура
    - Хорошо работает со стандартными настройками
    - Устойчива к вариациям гиперпараметров
    """
    
    def __init__(self, model, config, device, use_amp=True, gradient_accumulation_steps=1):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Стандартная кросс-энтропия для DenseNet
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        
        # AdamW с умеренной регуляризацией
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing с warmup
        self.scheduler = self._create_scheduler()
        
        # Смешанная точность - используем только AMP, без ручного FP16
        self.use_amp = use_amp and device.type == 'cuda'
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Мониторинг
        self.memory_monitor = GPUMemoryMonitor(device)
        
        # История
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_time': [],
            'learning_rates': [],
            'memory_usage': []
        }
        
        config.create_dirs()
        
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
    
    def _create_scheduler(self):
        """Cosine annealing с warmup"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def warmup_lambda(epoch):
            if epoch < 3:  # 3 эпохи warmup
                return (epoch + 1) / 3
            else:
                progress = (epoch - 3) / (self.config.EPOCHS - 3)
                return 0.5 * (1 + np.cos(np.pi * progress))
        
        return LambdaLR(self.optimizer, lr_lambda=warmup_lambda)
    
    def train_epoch(self, train_loader):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        self.memory_monitor.start()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc='Training')):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # DenseNet использует channels_last для скорости
            if self.device.type == 'cuda':
                images = images.to(memory_format=torch.channels_last)
            
            # Использование AMP
            if self.use_amp:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    loss = loss / self.gradient_accumulation_steps
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss = loss / self.gradient_accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
            
            running_loss += loss.item() * self.gradient_accumulation_steps
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if batch_idx % 100 == 0:
                self.memory_monitor.step()
            
            if self.device.type == 'cuda' and batch_idx % 100 == 0:
                torch.cuda.empty_cache()
        
        if len(train_loader) % self.gradient_accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
        
        epoch_time = time.time() - start_time
        memory_stats = self.memory_monitor.stop()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        return train_loss, train_acc, epoch_time, memory_stats
    
    def validate(self, val_loader):
        """Валидация"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc='Validation'):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                if self.device.type == 'cuda':
                    images = images.to(memory_format=torch.channels_last)
                
                if self.use_amp:
                    with torch.amp.autocast('cuda'):
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
        
        print("=" * 70)
        print("DENSENET-121 TRAINING (GPU OPTIMIZED)")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        print(f"Total epochs: {epochs}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Weight decay: {self.config.WEIGHT_DECAY}")
        print("=" * 70)
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch [{epoch}/{epochs}]")
            print("-" * 50)
            
            train_loss, train_acc, epoch_time, memory_stats = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            
            self.scheduler.step()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['train_time'].append(epoch_time)
            self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
            
            if memory_stats:
                self.history['memory_usage'].append(memory_stats)
            
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s")
            
            if memory_stats:
                print(f"GPU Memory: {memory_stats.get('max_allocated_gb', 0):.2f} GB")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"*** New best model! Val Acc: {val_acc:.2f}% ***")
            
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_acc)
        
        print("\n" + "=" * 70)
        print(f"Training completed! Best Val Acc: {best_val_acc:.2f}%")
        print("=" * 70)
        
        return self.history
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Сохранение чекпоинта"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_acc': val_acc,
            'history': self.history
        }
        
        if is_best:
            path = self.config.CHECKPOINTS_PATH / 'best_model.pth'
        else:
            path = self.config.CHECKPOINTS_PATH / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def save_history(self):
        """Сохранение истории"""
        history_path = self.config.LOGS_PATH / 'training_history.json'
        
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (np.ndarray, np.generic)):
                    history_serializable[key] = [v.item() if isinstance(v, np.generic) else v for v in value]
                elif isinstance(value[0], dict):
                    history_serializable[key] = value
                else:
                    history_serializable[key] = value
            else:
                history_serializable[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=4, default=str)
        
        self.plot_history()
    
    def plot_history(self):
        """Построение графиков"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        
        ax1.plot(self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(self.history['val_loss'], label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.history['train_acc'], label='Train Acc', linewidth=2)
        ax2.plot(self.history['val_acc'], label='Val Acc', linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.plot(self.history['learning_rates'], label='Learning Rate', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.plot(self.history['train_time'], label='Epoch Time', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Training Time per Epoch')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_PATH / 'training_history.png', dpi=150)
        plt.show()

def evaluate_densenet(model, test_loader, config, device, use_amp=True):
    """Оценка модели"""
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if device.type == 'cuda':
                images = images.to(memory_format=torch.channels_last)
            
            if use_amp:
                with torch.amp.autocast('cuda'):
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
    
    results_path = config.RESULTS_PATH / 'test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - DenseNet-121')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(config.RESULTS_PATH / 'confusion_matrix.png', dpi=150)
    plt.show()
    
    # Анализ ошибок
    errors = []
    for i, (pred, true) in enumerate(zip(all_preds, all_labels)):
        if pred != true:
            errors.append((true, pred))
    
    if errors:
        print(f"\nTotal errors: {len(errors)} out of {len(all_labels)}")
        print(f"Error rate: {len(errors) / len(all_labels) * 100:.2f}%")
        
        error_counts = Counter(errors)
        print("\nMost common confusions:")
        for (true, pred), count in error_counts.most_common(10):
            print(f"  Class {true} -> Class {pred}: {count} times")
    
    return results, all_preds, all_labels

def main():
    """Запуск обучения"""
    config = DenseNetConfig
    
    gpu_manager = GPUManager()
    device = gpu_manager.device
    gpu_manager.print_gpu_info()
    
    print("\nLoading data with GPU optimization...")
    train_loader, val_loader, test_loader = get_densenet_data_loaders(config, device)
    print(f"Train: {len(train_loader.dataset)} samples")
    print(f"Val: {len(val_loader.dataset)} samples")
    print(f"Test: {len(test_loader.dataset)} samples")
    
    print("\nCreating model...")
    model = create_densenet121_gpu(config, device)
    
    print("\nStarting training...")
    trainer = DenseNetTrainer(
        model, 
        config, 
        device, 
        use_amp=True,
        gradient_accumulation_steps=1
    )
    history = trainer.train(train_loader, val_loader)
    
    trainer.save_history()
    
    print("\nEvaluating on test set...")
    results, preds, labels = evaluate_densenet(model, test_loader, config, device, use_amp=True)
    
    print("\nTest Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}")
    
    print("\nProfiling model performance...")
    perf_stats = profile_model_performance(model, device)
    print(f"Average inference time: {perf_stats['avg_inference_time_ms']:.2f} ms")
    print(f"FPS: {perf_stats['fps']:.2f}")
    
    # Сохранение модели
    final_model_path = config.RUNS_PATH / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"\nModel saved to: {final_model_path}")
    
    # Генерация отчета
    generate_report(config, model, results, history, perf_stats)

def generate_report(config, model, results, history, perf_stats):
    """Генерация отчета"""
    report = {
        'model_name': config.MODEL_NAME,
        'architecture': 'DenseNet-121',
        'input_size': config.INPUT_SIZE,
        'num_classes': config.NUM_CLASSES,
        'batch_size': config.BATCH_SIZE,
        'epochs': config.EPOCHS,
        'learning_rate': config.LEARNING_RATE,
        'weight_decay': config.WEIGHT_DECAY,
        'total_params': model.get_total_params(),
        'trainable_params': model.get_trainable_params(),
        'model_size_mb': model.get_model_size_mb(),
        'best_val_acc': max(history['val_acc']),
        'test_accuracy': results['accuracy'],
        'test_precision': results['precision'],
        'test_recall': results['recall'],
        'test_f1': results['f1_score'],
        'inference_time_ms': perf_stats.get('avg_inference_time_ms', 0),
        'fps': perf_stats.get('fps', 0),
        'training_time': sum(history['train_time']),
        'best_epoch': np.argmax(history['val_acc']) + 1,
        'gpu_info': {
            'device': str(config.DEVICE),
            'mixed_precision': True
        }
    }
    
    report_path = config.RESULTS_PATH / 'model_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print("\n" + "=" * 70)
    print("SUMMARY REPORT - DENSENET-121")
    print("=" * 70)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()