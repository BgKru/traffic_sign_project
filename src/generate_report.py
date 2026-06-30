# src/generate_report.py

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import base64
from io import BytesIO

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PDFReport(FPDF):
    """PDF отчет с результатами сравнения"""
    
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'Model Comparison Report', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
        self.ln(5)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)
    
    def chapter_body(self, body):
        self.set_font('Arial', '', 11)
        self.multi_cell(0, 7, body)
        self.ln()
    
    def add_table(self, data, headers=None):
        """Добавление таблицы в PDF"""
        if headers:
            self.set_font('Arial', 'B', 10)
            col_widths = [30] + [25] * (len(headers) - 1)
            
            # Заголовки
            for i, header in enumerate(headers):
                self.cell(col_widths[i], 10, header, 1, 0, 'C')
            self.ln()
        
        self.set_font('Arial', '', 9)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 8, str(cell), 1, 0, 'C')
            self.ln()
        self.ln()

def generate_pdf_report(results_dir='comparison_results'):
    """
    Генерация PDF отчета
    """
    results_dir = Path(results_dir)
    results_path = results_dir / 'comparison_results.json'
    
    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        return
    
    # Загрузка результатов
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Создание PDF
    pdf = PDFReport()
    pdf.add_page()
    
    # Заголовок
    pdf.chapter_title('Executive Summary')
    pdf.chapter_body(
        f'This report presents a comprehensive comparison of 5 different neural network '
        f'architectures for traffic sign classification on the GTSRB dataset. '
        f'The models were evaluated on accuracy, precision, recall, F1-score, '
        f'inference speed, and model size.'
    )
    
    # Основные выводы
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    fastest_model = max(results.items(), key=lambda x: x[1]['fps'])
    
    pdf.chapter_title('Key Findings')
    pdf.chapter_body(
        f'• Best Accuracy: {best_model[0]} ({best_model[1]["accuracy"]*100:.2f}%)\n'
        f'• Fastest Model: {fastest_model[0]} ({fastest_model[1]["fps"]:.1f} FPS)\n'
        f'• Dataset: GTSRB (43 classes, 50,000+ images)'
    )
    
    # Таблица сравнения
    pdf.chapter_title('Detailed Comparison')
    
    headers = ['Model', 'Accuracy%', 'F1%', 'FPS', 'Size(MB)']
    table_data = []
    
    for model_name, model_results in sorted(results.items(), 
                                           key=lambda x: x[1]['accuracy'], 
                                           reverse=True):
        row = [
            model_name,
            f'{model_results["accuracy"]*100:.2f}',
            f'{model_results["f1_score"]*100:.2f}',
            f'{model_results["fps"]:.1f}',
            f'{model_results.get("model_size_mb", 0):.1f}'
        ]
        table_data.append(row)
    
    pdf.add_table(table_data, headers)
    
    # Детальный анализ каждой модели
    for model_name, model_results in results.items():
        pdf.add_page()
        pdf.chapter_title(f'Model: {model_name}')
        
        stats = [
            f'Accuracy: {model_results["accuracy"]*100:.2f}%',
            f'Precision: {model_results["precision"]*100:.2f}%',
            f'Recall: {model_results["recall"]*100:.2f}%',
            f'F1-Score: {model_results["f1_score"]*100:.2f}%',
            f'Error Rate: {model_results["error_rate"]:.2f}%',
            f'Inference Time: {model_results["avg_inference_time_ms"]:.2f} ms',
            f'FPS: {model_results["fps"]:.1f}',
            f'Total Errors: {model_results["total_errors"]}/{model_results["total_samples"]}',
            f'Avg Confidence: {model_results["avg_confidence"]*100:.2f}%'
        ]
        
        pdf.chapter_body('\n'.join(stats))
        
        # Добавление примеров ошибок
        if model_results.get('error_examples'):
            pdf.chapter_title('Common Errors')
            error_text = ''
            for error in model_results['error_examples'][:10]:
                error_text += f"• True: {error['true_label']} → Pred: {error['pred_label']} (Conf: {error['confidence']*100:.1f}%)\n"
            pdf.chapter_body(error_text)
    
    # Рекомендации
    pdf.add_page()
    pdf.chapter_title('Recommendations')
    
    recommendations = {
        'Production Use': f'{best_model[0]} - Best accuracy for production systems',
        'Real-Time Applications': f'{fastest_model[0]} - Highest FPS for real-time inference',
        'Edge Devices': f'{min(results.items(), key=lambda x: x[1].get("model_size_mb", float("inf")))[0]} - Smallest model size'
    }
    
    for use_case, recommendation in recommendations.items():
        pdf.chapter_body(f'{use_case}: {recommendation}')
    
    # Сохранение
    pdf_path = results_dir / 'comparison_report.pdf'
    pdf.output(str(pdf_path))
    print(f"PDF report saved to: {pdf_path}")

if __name__ == '__main__':
    generate_pdf_report()