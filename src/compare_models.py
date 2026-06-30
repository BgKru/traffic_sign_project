# src/compare_models.py (исправленная версия)

import os
import sys
import json
import time
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Импорт конфигураций
from configs.resnet50_config import ResNet50Config
from configs.efficientnet_config import EfficientNetConfig
from configs.mobilenet_config import MobileNetV3Config
from configs.vit_config import ViTConfig
from configs.densenet_config import DenseNetConfig

# Импорт моделей
from src.resnet.model_resnet50 import ResNet50Classifier
from src.efficientnet.model_efficientnet import EfficientNetB0Classifier
from src.mobilenet.model_mobilenet import MobileNetV3LargeGPU
from src.vit.model_vit import VisionTransformerGPU
from src.densent.model_densenet import DenseNet121GPU

# Импорт загрузчиков данных
from src.data_loader import get_data_loaders as get_standard_loaders
from src.mobilenet.data_loader_mobilenet import get_mobilenet_data_loaders
from src.vit.data_loader_vit import get_vit_data_loaders
from src.densent.data_loader_densenet import get_densenet_data_loaders

# Импорт GPU утилит
from src.utils import GPUManager

def get_model_class(model_name):
    """
    Возвращает класс модели по имени
    """
    model_map = {
        'ResNet-50': ResNet50Classifier,
        'EfficientNet-B0': EfficientNetB0Classifier,
        'MobileNetV3-Large': MobileNetV3LargeGPU,
        'ViT-B/16': VisionTransformerGPU,
        'DenseNet-121': DenseNet121GPU
    }
    return model_map.get(model_name)

def get_model_config(model_name):
    """
    Возвращает конфигурацию модели по имени
    """
    config_map = {
        'ResNet-50': ResNet50Config,
        'EfficientNet-B0': EfficientNetConfig,
        'MobileNetV3-Large': MobileNetV3Config,
        'ViT-B/16': ViTConfig,
        'DenseNet-121': DenseNetConfig
    }
    return config_map.get(model_name)

def get_loader_for_model(model_name, config, device):
    """
    Получение DataLoader для конкретной модели
    """
    if 'MobileNet' in model_name:
        from src.mobilenet.data_loader_mobilenet import get_mobilenet_data_loaders
        return get_mobilenet_data_loaders(config, device)
    elif 'ViT' in model_name:
        from src.vit.data_loader_vit import get_vit_data_loaders
        return get_vit_data_loaders(config, device)
    elif 'DenseNet' in model_name:
        from src.densent.data_loader_densenet import get_densenet_data_loaders
        return get_densenet_data_loaders(config, device)
    else:
        from src.data_loader import get_data_loaders
        return get_data_loaders(config)

def find_best_weights(model_name, base_runs_path):
    """
    Поиск лучших весов для модели
    """
    runs_path = Path(base_runs_path)
    
    # Маппинг имен моделей к папкам
    model_folder_map = {
        'ResNet-50': 'resnet50',
        'EfficientNet-B0': 'efficientnet_b0',
        'MobileNetV3-Large': 'mobilenetv3_large_gpu',
        'ViT-B/16': 'vit_base_patch16_224_gpu',
        'DenseNet-121': 'densenet121_gpu'
    }
    
    folder_name = model_folder_map.get(model_name)
    if not folder_name:
        return None
    
    model_path = runs_path / folder_name
    
    # Проверяем возможные пути
    possible_paths = [
        model_path / 'checkpoints' / 'best_model.pth',
        model_path / 'final_model.pth',
        model_path / 'best_model.pth',
        runs_path / 'resnet50' / 'checkpoints' / 'best_model.pth',
        runs_path / 'efficientnet_b0' / 'checkpoints' / 'best_model.pth',
        runs_path / 'mobilenetv3_large' / 'checkpoints' / 'best_model.pth',
        runs_path / 'vit_base_patch16_224' / 'checkpoints' / 'best_model.pth',
        runs_path / 'densenet121' / 'checkpoints' / 'best_model.pth',
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    return None

class ModelComparator:
    """
    Класс для сравнения всех архитектур
    """
    
    def __init__(self, device=None):
        self.gpu_manager = GPUManager(device)
        self.device = self.gpu_manager.device
        self.gpu_manager.print_gpu_info()
        
        self.results = {}
        self.models_info = {}
        
        # Создание директории для результатов
        self.results_dir = Path(__file__).parent.parent / 'comparison_results'
        self.results_dir.mkdir(exist_ok=True)
    
    def load_model(self, model_class, weights_path, config, model_name):
        """
        Загрузка модели из сохраненных весов
        """
        print(f"\nLoading {model_name}...")
        
        # Создание модели с единым интерфейсом
        try:
            model = model_class(
                num_classes=config.NUM_CLASSES,
                pretrained=False,
                freeze_backbone=False
            )
        except TypeError:
            # Fallback если параметр freeze_backbone не поддерживается
            model = model_class(
                num_classes=config.NUM_CLASSES,
                pretrained=False
            )
        
        # Загрузка весов
        if weights_path and os.path.exists(weights_path):
            try:
                checkpoint = torch.load(weights_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"  Loaded checkpoint with val_acc: {checkpoint.get('val_acc', 'N/A')}")
                else:
                    model.load_state_dict(checkpoint)
                    print(f"  Loaded state_dict only")
            except Exception as e:
                print(f"  Error loading weights: {e}")
                print(f"  Using randomly initialized model")
        else:
            print(f"  Weights not found at {weights_path}")
            print(f"  Using randomly initialized model")
        
        model = model.to(self.device)
        model.eval()
        
        # Информация о модели
        try:
            total_params = model.get_total_params()
            trainable_params = model.get_trainable_params()
            model_size = model.get_model_size_mb()
        except:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size = total_params * 4 / (1024 * 1024)
        
        self.models_info[model_name] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size
        }
        
        return model
    
    def evaluate_model(self, model, test_loader, model_name):
        """
        Оценка модели на тестовых данных
        """
        print(f"\nEvaluating {model_name}...")
        
        model.eval()
        all_preds = []
        all_labels = []
        all_confidences = []
        inference_times = []
        error_examples = []
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(tqdm(test_loader, desc=f'Testing {model_name}')):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                # Измерение времени инференса
                start_time = time.time()
                outputs = model(images)
                inference_time = (time.time() - start_time) * 1000
                
                inference_times.append(inference_time / images.size(0))
                
                probs = torch.softmax(outputs, dim=1)
                confidences, preds = torch.max(probs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidences.cpu().numpy())
                
                # Сохранение ошибок
                if len(error_examples) < 50:
                    for i in range(min(images.size(0), 50 - len(error_examples))):
                        if preds[i] != labels[i]:
                            error_examples.append({
                                'true_label': labels[i].item(),
                                'pred_label': preds[i].item(),
                                'confidence': confidences[i].item()
                            })
        
        # Вычисление метрик
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        cm = confusion_matrix(all_labels, all_preds)
        
        total_errors = sum(1 for p, l in zip(all_preds, all_labels) if p != l)
        error_rate = total_errors / len(all_labels) * 100
        
        avg_inference_time = np.mean(inference_times) if inference_times else 0
        std_inference_time = np.std(inference_times) if inference_times else 0
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'error_rate': error_rate,
            'total_errors': total_errors,
            'total_samples': len(all_labels),
            'avg_confidence': np.mean(all_confidences) if all_confidences else 0,
            'avg_inference_time_ms': avg_inference_time,
            'std_inference_time_ms': std_inference_time,
            'fps': 1000 / avg_inference_time if avg_inference_time > 0 else 0,
            'confusion_matrix': cm.tolist(),
            'error_examples': error_examples[:20]
        }
        
        self.results[model_name] = results
        
        print(f"\n{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  Error Rate: {error_rate:.2f}%")
        print(f"  Avg Inference Time: {avg_inference_time:.2f} ms")
        print(f"  FPS: {1000 / avg_inference_time:.2f}")
        
        return results
    
    def compare_all_models(self):
        """
        Сравнение всех 5 моделей
        """
        print("\n" + "=" * 70)
        print("STARTING COMPARATIVE ANALYSIS OF 5 ARCHITECTURES")
        print("=" * 70)
        
        base_path = Path(__file__).parent.parent
        runs_path = base_path / 'runs'
        
        # Список моделей для сравнения
        model_names = ['ResNet-50', 'EfficientNet-B0', 'MobileNetV3-Large', 'ViT-B/16', 'DenseNet-121']
        
        for model_name in model_names:
            print(f"\n{'='*50}")
            print(f"Processing: {model_name}")
            print('='*50)
            
            # Получение класса модели
            model_class = get_model_class(model_name)
            config = get_model_config(model_name)
            
            if not model_class or not config:
                print(f"  Model {model_name} not found, skipping...")
                continue
            
            # Поиск весов
            weights_path = find_best_weights(model_name, runs_path)
            print(f"  Weights path: {weights_path}")
            
            # Загрузка модели
            model = self.load_model(model_class, weights_path, config, model_name)
            
            # Загрузка данных
            train_loader, val_loader, test_loader = get_loader_for_model(model_name, config, self.device)
            print(f"  Test dataset size: {len(test_loader.dataset)}")
            
            # Оценка модели
            self.evaluate_model(model, test_loader, model_name)
        
        # Сохранение и визуализация результатов
        self.save_results()
        self.generate_comparison_table()
        self.generate_visualizations()
        self.generate_summary()
        
        return self.results
    
    def save_results(self):
        """Сохранение результатов в JSON"""
        save_data = {}
        for model_name, results in self.results.items():
            model_info = self.models_info.get(model_name, {})
            save_data[model_name] = {
                'accuracy': float(results['accuracy']),
                'precision': float(results['precision']),
                'recall': float(results['recall']),
                'f1_score': float(results['f1_score']),
                'error_rate': float(results['error_rate']),
                'total_errors': int(results['total_errors']),
                'total_samples': int(results['total_samples']),
                'avg_confidence': float(results['avg_confidence']),
                'avg_inference_time_ms': float(results['avg_inference_time_ms']),
                'fps': float(results['fps']),
                'total_params': int(model_info.get('total_params', 0)),
                'model_size_mb': float(model_info.get('model_size_mb', 0))
            }
        
        results_path = self.results_dir / 'comparison_results.json'
        with open(results_path, 'w') as f:
            json.dump(save_data, f, indent=4)
        
        print(f"\nResults saved to: {results_path}")
    
    def generate_comparison_table(self):
        """Генерация таблицы сравнения"""
        data = []
        for model_name, results in self.results.items():
            model_info = self.models_info.get(model_name, {})
            
            row = {
                'Model': model_name,
                'Accuracy (%)': results['accuracy'] * 100,
                'Precision (%)': results['precision'] * 100,
                'Recall (%)': results['recall'] * 100,
                'F1-Score (%)': results['f1_score'] * 100,
                'Error Rate (%)': results['error_rate'],
                'Inference (ms)': results['avg_inference_time_ms'],
                'FPS': results['fps'],
                'Size (MB)': model_info.get('model_size_mb', 0)
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        df = df.sort_values('Accuracy (%)', ascending=False)
        
        # Сохранение в CSV
        csv_path = self.results_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False)
        print(f"\nComparison table saved to: {csv_path}")
        
        # Вывод таблицы
        print("\n" + "=" * 90)
        print("COMPARISON TABLE")
        print("=" * 90)
        print(df.to_string(index=False))
        
        return df
    
    def generate_visualizations(self):
        """Генерация визуализаций"""
        if not self.results:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        model_names = list(self.results.keys())
        
        # 1. Accuracy
        acc_values = [self.results[m]['accuracy'] * 100 for m in model_names]
        ax = axes[0, 0]
        colors = plt.cm.RdYlGn(np.array(acc_values) / max(acc_values))
        bars = ax.bar(model_names, acc_values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Accuracy Comparison')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, acc_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}%', ha='center', va='bottom')
        
        # 2. F1-Score
        f1_values = [self.results[m]['f1_score'] * 100 for m in model_names]
        ax = axes[0, 1]
        colors = plt.cm.RdYlGn(np.array(f1_values) / max(f1_values))
        bars = ax.bar(model_names, f1_values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('F1-Score (%)')
        ax.set_title('F1-Score Comparison')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, f1_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}%', ha='center', va='bottom')
        
        # 3. Inference Speed (FPS)
        fps_values = [self.results[m]['fps'] for m in model_names]
        ax = axes[0, 2]
        colors = plt.cm.Blues(np.array(fps_values) / max(fps_values) if max(fps_values) > 0 else np.ones(len(fps_values)))
        bars = ax.bar(model_names, fps_values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('FPS')
        ax.set_title('Inference Speed (FPS)')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, fps_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom')
        
        # 4. Model Size
        size_values = [self.models_info.get(m, {}).get('model_size_mb', 0) for m in model_names]
        ax = axes[1, 0]
        colors = plt.cm.Reds(np.array(size_values) / max(size_values) if max(size_values) > 0 else np.ones(len(size_values)))
        bars = ax.bar(model_names, size_values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Model Size')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, size_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.1f}', ha='center', va='bottom')
        
        # 5. Error Rate
        error_values = [self.results[m]['error_rate'] for m in model_names]
        ax = axes[1, 1]
        colors = plt.cm.RdYlGn(np.array(error_values) / max(error_values) if max(error_values) > 0 else np.ones(len(error_values)))
        bars = ax.bar(model_names, error_values, color=colors)
        ax.set_xlabel('Model')
        ax.set_ylabel('Error Rate (%)')
        ax.set_title('Error Rate')
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        for bar, val in zip(bars, error_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                   f'{val:.2f}%', ha='center', va='bottom')
        
        # 6. Accuracy vs Speed Scatter
        ax = axes[1, 2]
        for model_name in model_names:
            acc = self.results[model_name]['accuracy'] * 100
            fps = self.results[model_name]['fps']
            size = self.models_info.get(model_name, {}).get('model_size_mb', 0)
            ax.scatter(acc, fps, s=size * 5, alpha=0.6, label=model_name)
            ax.annotate(model_name, (acc, fps), xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('Accuracy (%)')
        ax.set_ylabel('FPS')
        ax.set_title('Accuracy vs Speed (Size = dot size)')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        # Сохранение
        fig_path = self.results_dir / 'comparison_visualizations.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"\nVisualizations saved to: {fig_path}")
        plt.show()
    
    def generate_summary(self):
        """Генерация текстового резюме"""
        if not self.results:
            return
        
        best_model = max(self.results.items(), key=lambda x: x[1]['accuracy'])
        fastest_model = max(self.results.items(), key=lambda x: x[1]['fps'])
        smallest_model = min(self.models_info.items(), key=lambda x: x[1].get('model_size_mb', float('inf')))
        
        summary = f"""
================================================================================
                            COMPARISON SUMMARY
================================================================================

Dataset: GTSRB (German Traffic Sign Recognition Benchmark)
Number of Classes: 43
Device: {self.device}

--------------------------------------------------------------------------------
KEY FINDINGS
--------------------------------------------------------------------------------

🏆 Best Accuracy: {best_model[0]} ({best_model[1]['accuracy']*100:.2f}%)
   - F1-Score: {best_model[1]['f1_score']*100:.2f}%
   - Error Rate: {best_model[1]['error_rate']:.2f}%

⚡ Fastest Model: {fastest_model[0]} ({fastest_model[1]['fps']:.1f} FPS)
   - Inference Time: {fastest_model[1]['avg_inference_time_ms']:.2f} ms

💾 Smallest Model: {smallest_model[0]} ({smallest_model[1].get('model_size_mb', 0):.1f} MB)

--------------------------------------------------------------------------------
MODEL RANKINGS
--------------------------------------------------------------------------------
"""
        
        # Сортировка по точности
        sorted_models = sorted(self.results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (model_name, results) in enumerate(sorted_models, 1):
            size = self.models_info.get(model_name, {}).get('model_size_mb', 0)
            summary += f"{i}. {model_name}\n"
            summary += f"   Accuracy: {results['accuracy']*100:.2f}% | "
            summary += f"F1: {results['f1_score']*100:.2f}% | "
            summary += f"FPS: {results['fps']:.1f} | "
            summary += f"Size: {size:.1f} MB\n"
        
        summary += f"""
--------------------------------------------------------------------------------
RECOMMENDATIONS
--------------------------------------------------------------------------------

• For Production (Best Accuracy): {best_model[0]}
• For Real-Time Applications: {fastest_model[0]}
• For Edge Devices: {smallest_model[0]}

================================================================================
"""
        
        # Сохранение
        summary_path = self.results_dir / 'comparison_summary.txt'
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(summary)
        print(f"Summary saved to: {summary_path}")

def main():
    """Основная функция"""
    comparator = ModelComparator()
    results = comparator.compare_all_models()
    
    print("\n" + "=" * 70)
    print("COMPARISON COMPLETE!")
    print("=" * 70)
    print(f"Results saved to: {comparator.results_dir}")

if __name__ == '__main__':
    main()