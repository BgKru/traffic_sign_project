import os
import sys
import json
import time
import multiprocessing as mp
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.cuda.amp import GradScaler, autocast
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
from datetime import datetime

# Добавляем пути для импорта
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from configs.efficientnet_config import EfficientNetConfig
from src.data_loader import get_data_loaders
from src.efficientnet.model_efficientnet import create_efficientnet_b0

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
    
class EfficientNetTrainer:
    """Класс для обучения EfficientNet-B0"""
    
    def __init__(self, model, config, device='cuda', use_amp=True):
        """
        Args:
            model: Модель PyTorch
            config: Объект конфигурации
            device: Устройство для обучения
            use_amp: Использовать автоматическое смешанное обучение
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.use_amp = use_amp and device.type == 'cuda'
        
        # Потери и оптимизатор
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
            betas=(0.9, 0.999)
        )
        
        # Планировщик learning rate с cosine annealing
        self.scheduler = lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.EPOCHS,
            eta_min=1e-6
        )
        
        # GradScaler для смешанной точности
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # История метрик
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'train_time': [],
            'learning_rates': []
        }
        
        # Создание директорий
        config.create_dirs()
        
        # Оптимизация для GPU
        if device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            print(f"CUDA optimizations enabled")
        
    def train_epoch(self, train_loader):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (images, labels) in enumerate(tqdm(train_loader, desc='Training')):
            # Перемещение данных на устройство
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)
            
            # Обнуление градиентов
            self.optimizer.zero_grad()
            
            # Forward pass с автоматическим смешанным обучением
            with autocast(enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            
            # Backward pass с масштабированием градиентов
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Статистика
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Очистка памяти (периодическая)
            if self.device.type == 'cuda' and batch_idx % 50 == 0:
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
                
                with autocast(enabled=self.use_amp):
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
        print("EFFICIENTNET-B0 TRAINING")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Total epochs: {epochs}")
        print(f"Batch size: {self.config.BATCH_SIZE}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Weight decay: {self.config.WEIGHT_DECAY}")
        print("=" * 70)
        
        best_val_acc = 0.0
        
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch [{epoch}/{epochs}]")
            print("-" * 50)
            
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
            
            # Сохранение лучшей модели
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_checkpoint(epoch, val_acc, is_best=True)
                print(f"*** New best model! Val Acc: {val_acc:.2f}% ***")
            
            # Регулярное сохранение
            if epoch % 5 == 0:
                self.save_checkpoint(epoch, val_acc)
        
        print("\n" + "=" * 70)
        print(f"Training completed! Best Val Acc: {best_val_acc:.2f}%")
        print("=" * 70)
        
        return self.history
    
    def save_checkpoint(self, epoch, val_acc, is_best=False):
        """Сохранение чекпоинта модели"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.use_amp else None,
            'val_acc': val_acc,
            'history': self.history,
            'config': {
                'model_name': self.config.MODEL_NAME,
                'input_size': self.config.INPUT_SIZE,
                'num_classes': self.config.NUM_CLASSES,
                'batch_size': self.config.BATCH_SIZE
            }
        }
        
        if is_best:
            path = self.config.CHECKPOINTS_PATH / 'best_model.pth'
        else:
            path = self.config.CHECKPOINTS_PATH / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")
        
    def save_history(self):
        """Сохранение истории обучения"""
        history_path = self.config.LOGS_PATH / 'training_history.json'
        
        # Конвертация numpy для JSON
        history_serializable = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                history_serializable[key] = [float(v) if isinstance(v, (np.floating, float)) else v for v in value]
            else:
                history_serializable[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(history_serializable, f, indent=4)
        
        print(f"History saved to: {history_path}")
        
        # Сохранение графиков
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
        
        # График времени обучения
        ax4.plot(self.history['train_time'], label='Epoch Time', linewidth=2)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Training Time per Epoch')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.config.RESULTS_PATH / 'training_history.png', dpi=150)
        plt.show()

def evaluate_efficientnet(model, test_loader, config, device='cuda', use_amp=True):
    """Оценка модели на тестовых данных"""
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Метрики
    test_loss /= len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    # Матрица ошибок
    cm = confusion_matrix(all_labels, all_preds)
    
    # Сохранение результатов
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
    
    print(f"Results saved to: {results_path}")
    
    # График матрицы ошибок
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.get_class_names(), 
                yticklabels=config.get_class_names())
    plt.title('Confusion Matrix - EfficientNet-B0')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.RESULTS_PATH / 'confusion_matrix.png', dpi=150)
    plt.show()
    
    return results, all_preds, all_labels

def measure_inference_time(model, device, input_size=(1, 3, 224, 224), num_runs=100):
    """
    Измерение времени инференса модели
    """
    model.eval()
    model.to(device)
    
    # Создание случайного входного тензора
    x = torch.randn(input_size).to(device)
    
    # Прогрев (warmup)
    print("Warming up...")
    for _ in range(20):
        with torch.no_grad():
            _ = model(x)
    
    # Синхронизация для точного измерения на GPU
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Измерение времени
    print(f"Measuring inference time ({num_runs} runs)...")
    times = []
    for i in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        times.append((time.time() - start_time) * 1000)  # в миллисекундах
        
        # Прогресс
        if (i + 1) % 10 == 0:
            print(f"  Run {i+1}/{num_runs}")
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    min_time = np.min(times)
    max_time = np.max(times)
    
    print(f"\nInference time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"Range: [{min_time:.2f}, {max_time:.2f}] ms")
    print(f"FPS: {1000 / avg_time:.2f}")
    
    return avg_time, std_time

def main():
    """Основная функция для запуска обучения"""
    
    print("=" * 70)
    print("EFFICIENTNET-B0 TRAINING SCRIPT")
    print("=" * 70)
    
    # Инициализация конфигурации
    config = EfficientNetConfig
    
    # Устройство
    device = torch.device(config.DEVICE if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Загрузка данных
    print("\n" + "=" * 70)
    print("LOADING DATA")
    print("=" * 70)
    train_loader, val_loader, test_loader = get_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    
    # Проверка распределения классов
    print("\nClass distribution (first 5 classes):")
    class_counts = train_loader.dataset.class_counts
    for class_id in sorted(class_counts.keys())[:5]:
        print(f"  Class {class_id}: {class_counts[class_id]} samples")
    
    # Создание модели
    print("\n" + "=" * 70)
    print("CREATING MODEL")
    print("=" * 70)
    model = create_efficientnet_b0(config, device=device)
    
    # Обучение
    print("\n" + "=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)
    use_amp = True if device.type == 'cuda' else False
    trainer = EfficientNetTrainer(model, config, device, use_amp=use_amp)
    history = trainer.train(train_loader, val_loader)
    
    # Сохранение истории
    print("\n" + "=" * 70)
    print("SAVING HISTORY")
    print("=" * 70)
    trainer.save_history()
    
    # Оценка на тестовых данных
    print("\n" + "=" * 70)
    print("EVALUATING ON TEST SET")
    print("=" * 70)
    results, preds, labels = evaluate_efficientnet(model, test_loader, config, device, use_amp=use_amp)
    
    print("\nTest Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall: {results['recall']:.4f}")
    print(f"  F1-Score: {results['f1_score']:.4f}")
    print(f"  Test Loss: {results['test_loss']:.4f}")
    
    # Измерение времени инференса
    print("\n" + "=" * 70)
    print("MEASURING INFERENCE TIME")
    print("=" * 70)
    avg_time, std_time = measure_inference_time(model, device)
    
    # Сохранение финальной модели
    print("\n" + "=" * 70)
    print("SAVING FINAL MODEL")
    print("=" * 70)
    final_model_path = config.RUNS_PATH / 'final_model.pth'
    torch.save(model.state_dict(), final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    # Генерация отчета
    print("\n" + "=" * 70)
    print("GENERATING REPORT")
    print("=" * 70)
    generate_report(config, model, results, history, avg_time)
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)

def generate_report(config, model, results, history, inference_time):
    """Генерация краткого отчета"""
    report = {
        'model_name': config.MODEL_NAME,
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
        'inference_time_ms': inference_time,
        'fps': 1000 / inference_time,
        'training_time_seconds': sum(history['train_time']),
        'best_epoch': np.argmax(history['val_acc']) + 1,
        'device': str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
    }
    
    report_path = config.RESULTS_PATH / 'model_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=4)
    
    print(f"\nReport saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("SUMMARY REPORT - EfficientNet-B0")
    print("=" * 70)
    for key, value in report.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()