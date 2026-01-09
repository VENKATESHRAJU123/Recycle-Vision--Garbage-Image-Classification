"""
Model evaluation and performance analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score
)
from tensorflow import keras
import json
from config import *

def evaluate_model(model, test_generator, model_name='model'):
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained Keras model
        test_generator: Test data generator
        model_name: Name for saving results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*60}")
    print(f"EVALUATING {model_name.upper()}")
    print(f"{'='*60}\n")
    
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    
    # Get true labels
    y_true = test_generator.classes
    
    # Get class names
    class_names = list(test_generator.class_indices.keys())
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)
    
    # Save classification report
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    report_path = os.path.join(OUTPUTS_DIR, 'reports', f'{model_name}_classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"\n✓ Report saved to: {report_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names, model_name)
    
    # Per-class accuracy
    plot_per_class_metrics(report_dict, model_name)
    
    # Prediction confidence analysis
    analyze_prediction_confidence(predictions, y_true, y_pred, class_names, model_name)
    
    # Misclassification analysis
    analyze_misclassifications(test_generator, y_true, y_pred, predictions, class_names, model_name)
    
    # Return metrics
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm.tolist(),
        'classification_report': report_dict
    }
    
    return results

def plot_confusion_matrix(cm, class_names, model_name):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(10, 8))
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})
    
    plt.title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, 'visualizations', f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Confusion matrix saved to: {save_path}")

def plot_per_class_metrics(report_dict, model_name):
    """Plot per-class precision, recall, and F1-score"""
    # Extract metrics
    classes = [k for k in report_dict.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
    
    metrics = {
        'Precision': [report_dict[c]['precision'] for c in classes],
        'Recall': [report_dict[c]['recall'] for c in classes],
        'F1-Score': [report_dict[c]['f1-score'] for c in classes]
    }
    
    # Create DataFrame
    df_metrics = pd.DataFrame(metrics, index=classes)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    df_metrics.plot(kind='bar', ax=ax, width=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_title(f'Per-Class Metrics - {model_name}', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, 'visualizations', f'{model_name}_per_class_metrics.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Per-class metrics saved to: {save_path}")

def analyze_prediction_confidence(predictions, y_true, y_pred, class_names, model_name):
    """Analyze prediction confidence distribution"""
    # Get confidence scores
    confidence_scores = np.max(predictions, axis=1)
    
    # Separate correct and incorrect predictions
    correct_mask = (y_true == y_pred)
    correct_confidence = confidence_scores[correct_mask]
    incorrect_confidence = confidence_scores[~correct_mask]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram
    axes[0].hist(correct_confidence, bins=30, alpha=0.7, label='Correct', 
                 color='green', edgecolor='black')
    axes[0].hist(incorrect_confidence, bins=30, alpha=0.7, label='Incorrect', 
                 color='red', edgecolor='black')
    axes[0].set_title('Prediction Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Confidence Score', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot
    data_to_plot = [correct_confidence, incorrect_confidence]
    axes[1].boxplot(data_to_plot, labels=['Correct', 'Incorrect'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', edgecolor='black'),
                    medianprops=dict(color='red', linewidth=2))
    axes[1].set_title('Confidence Score Comparison', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Confidence Score', fontsize=12)
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, 'visualizations', f'{model_name}_confidence_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Confidence analysis saved to: {save_path}")
    print(f"\n  Average confidence (correct):   {np.mean(correct_confidence):.4f}")
    print(f"  Average confidence (incorrect): {np.mean(incorrect_confidence):.4f}")

def analyze_misclassifications(test_generator, y_true, y_pred, predictions, class_names, model_name):
    """Visualize most common misclassifications"""
    # Find misclassified samples
    misclassified_indices = np.where(y_true != y_pred)[0]
    
    if len(misclassified_indices) == 0:
        print("\n✓ No misclassifications! Perfect model!")
        return
    
    # Get file paths
    file_paths = test_generator.filepaths
    
    # Select random misclassified samples (max 12)
    num_samples = min(12, len(misclassified_indices))
    sample_indices = np.random.choice(misclassified_indices, num_samples, replace=False)
    
    # Plot
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle(f'Misclassified Examples - {model_name}', fontsize=16, fontweight='bold')
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(sample_indices):
            sample_idx = sample_indices[idx]
            
            # Load image
            img_path = file_paths[sample_idx]
            img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
            img_array = keras.utils.img_to_array(img) / 255.0
            
            # Get predictions
            true_label = class_names[y_true[sample_idx]]
            pred_label = class_names[y_pred[sample_idx]]
            confidence = predictions[sample_idx][y_pred[sample_idx]]
            
            # Display
            ax.imshow(img_array)
            ax.set_title(f'True: {true_label}\nPred: {pred_label} ({confidence:.2f})',
                        fontsize=10, color='red')
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, 'visualizations', f'{model_name}_misclassifications.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✓ Misclassifications saved to: {save_path}")
    print(f"  Total misclassified: {len(misclassified_indices)} / {len(y_true)}")

def compare_models(results_dict):
    """
    Compare multiple models
    
    Args:
        results_dict: Dictionary with model names as keys and results as values
    """
    # Create comparison DataFrame
    comparison_data = []
    for model_name, results in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': results['accuracy'],
            'Precision': results['precision'],
            'Recall': results['recall'],
            'F1-Score': results['f1_score']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('F1-Score', ascending=False)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON - TEST SET RESULTS")
    print("="*60)
    print(df_comparison.to_string(index=False))
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df_comparison))
    width = 0.2
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
    
    for i, metric in enumerate(metrics):
        offset = width * (i - 1.5)
        bars = ax.bar(x + offset, df_comparison[metric], width, 
                     label=metric, color=colors[i], edgecolor='black', linewidth=1.2)
        ax.bar_label(bars, fmt='%.3f', fontsize=8)
    
    ax.set_title('Model Performance Comparison (Test Set)', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_comparison['Model'], rotation=45, ha='right')
    ax.set_ylim([0, 1.05])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUTS_DIR, 'visualizations', 'model_comparison_test.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save comparison
    csv_path = os.path.join(OUTPUTS_DIR, 'reports', 'model_comparison_test.csv')
    df_comparison.to_csv(csv_path, index=False)
    print(f"\n✓ Comparison saved to: {csv_path}")
    
    # Find best model
    best_model = df_comparison.iloc[0]
    print(f"\n🏆 BEST MODEL: {best_model['Model']}")
    print(f"   F1-Score: {best_model['F1-Score']:.4f}")
    print(f"   Accuracy: {best_model['Accuracy']:.4f}")
    
    return df_comparison

if __name__ == "__main__":
    print("✓ Evaluation module loaded!")
