"""
Model training pipeline
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from preprocessing import create_data_generators
from model_builder import *
from project_config import *


def plot_training_history(history, model_name):
    """
    Plot training history
    
    Args:
        history: Training history object
        model_name: Name for saving plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Training History', fontsize=16, fontweight='bold')
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUTS_DIR, 'visualizations', f'{model_name}_training_history.png'), dpi=300)
    plt.show()

def train_model(model, model_name, train_generator, val_generator, epochs=EPOCHS):
    """
    Train a model
    
    Args:
        model: Keras model to train
        model_name: Name for saving
        train_generator: Training data generator
        val_generator: Validation data generator
        epochs: Number of epochs
        
    Returns:
        Training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Get callbacks
    callbacks = get_callbacks(model_name)
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model_path = os.path.join(MODELS_DIR, f'{model_name}_final.h5')
    model.save(model_path)
    print(f"\n✓ Model saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(MODELS_DIR, 'metadata', f'{model_name}_history.json')
    with open(history_path, 'w') as f:
        json.dump(history.history, f)
    
    # Plot training history
    plot_training_history(history, model_name)
    
    return history

def train_all_models():
    """
    Train all models and compare performance
    """
    # Create data generators
    print("Loading data...")
    train_gen, val_gen, test_gen = create_data_generators()
    
    # Dictionary to store results
    results = {}
    
    # 1. Train Baseline CNN
    print("\n" + "="*60)
    print("TRAINING BASELINE CNN")
    print("="*60)
    baseline_model = build_baseline_cnn()
    baseline_history = train_model(baseline_model, 'baseline_cnn', train_gen, val_gen, epochs=30)
    results['Baseline CNN'] = {
        'val_accuracy': max(baseline_history.history['val_accuracy']),
        'val_loss': min(baseline_history.history['val_loss'])
    }
    
    # 2. Train MobileNetV2
    print("\n" + "="*60)
    print("TRAINING MOBILENETV2")
    print("="*60)
    mobilenet_model = build_transfer_learning_model('MobileNetV2', trainable=False)
    mobilenet_history = train_model(mobilenet_model, 'mobilenetv2', train_gen, val_gen, epochs=EPOCHS)
    results['MobileNetV2'] = {
        'val_accuracy': max(mobilenet_history.history['val_accuracy']),
        'val_loss': min(mobilenet_history.history['val_loss'])
    }
    
    # 3. Train EfficientNetB0
    print("\n" + "="*60)
    print("TRAINING EFFICIENTNETB0")
    print("="*60)
    efficientnet_model = build_transfer_learning_model('EfficientNetB0', trainable=False)
    efficientnet_history = train_model(efficientnet_model, 'efficientnetb0', train_gen, val_gen, epochs=EPOCHS)
    results['EfficientNetB0'] = {
        'val_accuracy': max(efficientnet_history.history['val_accuracy']),
        'val_loss': min(efficientnet_history.history['val_loss'])
    }
    
    # 4. Train ResNet50
    print("\n" + "="*60)
    print("TRAINING RESNET50")
    print("="*60)
    resnet_model = build_transfer_learning_model('ResNet50', trainable=False)
    resnet_history = train_model(resnet_model, 'resnet50', train_gen, val_gen, epochs=EPOCHS)
    results['ResNet50'] = {
        'val_accuracy': max(resnet_history.history['val_accuracy']),
        'val_loss': min(resnet_history.history['val_loss'])
    }
    
    # Compare results
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    import pandas as pd
    df_results = pd.DataFrame(results).T
    df_results = df_results.sort_values('val_accuracy', ascending=False)
    print(df_results)
    
    # Save comparison
    df_results.to_csv(os.path.join(OUTPUTS_DIR, 'reports', 'model_comparison.csv'))
    
    # Find best model
    best_model_name = df_results.index[0]
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  Validation Accuracy: {df_results.loc[best_model_name, 'val_accuracy']:.4f}")
    
    return results

if __name__ == "__main__":
    # Create necessary directories
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    os.makedirs(os.path.join(MODELS_DIR, 'metadata'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_DIR, 'visualizations'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUTS_DIR, 'reports'), exist_ok=True)
    
    # Train all models
    results = train_all_models()
    
    print("\n✓ All models trained successfully!")
