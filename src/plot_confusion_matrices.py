# src/plot_confusion_matrices.py

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.resnet50_config import ResNet50Config
from configs.efficientnet_config import EfficientNetConfig
from configs.mobilenet_config import MobileNetV3Config
from configs.vit_config import ViTConfig
from configs.densenet_config import DenseNetConfig

def plot_all_confusion_matrices(results_dir='comparison_results'):
    """
    Построение матриц ошибок для всех моделей
    """
    results_dir = Path(results_dir)
    results_path = results_dir / 'comparison_results.json'
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        print("Please run compare_models.py first")
        return
    
    # Загрузка результатов
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Загрузка названий классов
    configs = {
        'ResNet-50': ResNet50Config,
        'EfficientNet-B0': EfficientNetConfig,
        'MobileNetV3-Large': MobileNetV3Config,
        'ViT-B/16': ViTConfig,
        'DenseNet-121': DenseNetConfig
    }
    
    class_names = ResNet50Config.get_class_names()
    
    # Создание фигуры с субплотами
    n_models = len(results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Получение матрицы ошибок
        cm = np.array(model_results.get('confusion_matrix', []))
        
        if cm.size == 0:
            ax.text(0.5, 0.5, 'No confusion matrix available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(model_name)
            continue
        
        # Ограничиваем размер для читаемости (показываем только первые 20 классов)
        if cm.shape[0] > 20:
            # Выбираем классы с наибольшим количеством ошибок
            error_counts = cm.sum(axis=1) - np.diag(cm)
            top_classes = np.argsort(error_counts)[-20:]
            cm = cm[np.ix_(top_classes, top_classes)]
            display_names = [class_names[i] for i in top_classes]
        else:
            display_names = class_names[:cm.shape[0]]
        
        # Нормализация по строкам
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)
        
        # Построение heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=display_names, yticklabels=display_names,
                   ax=ax, square=True, cbar_kws={'label': 'Fraction'})
        
        ax.set_title(f'{model_name}\nAccuracy: {model_results["accuracy"]*100:.2f}%')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        
        # Поворот меток
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Скрыть пустые субплоты
    for idx in range(len(results), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Сохранение
    save_path = results_dir / 'confusion_matrices_all_models.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrices saved to: {save_path}")
    plt.show()

def plot_metrics_radar(results_dir='comparison_results'):
    """
    Построение радарного графика для сравнения метрик
    """
    results_dir = Path(results_dir)
    results_path = results_dir / 'comparison_results.json'
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    # Загрузка результатов
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Подготовка данных
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#1B998B']
    
    for idx, (model_name, model_results) in enumerate(results.items()):
        values = [model_results[m] * 100 for m in metrics]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=model_name, color=colors[idx % len(colors)])
        ax.fill(angles, values, alpha=0.1, color=colors[idx % len(colors)])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 100)
    ax.set_title('Model Comparison Radar Chart', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    # Сохранение
    save_path = results_dir / 'radar_chart.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Radar chart saved to: {save_path}")
    plt.show()

if __name__ == '__main__':
    # Построение матриц ошибок
    plot_all_confusion_matrices()
    
    # Построение радарного графика
    plot_metrics_radar()