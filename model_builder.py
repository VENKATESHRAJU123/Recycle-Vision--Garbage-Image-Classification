"""
Deep learning model architectures for garbage classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    MobileNetV2, EfficientNetB0, ResNet50, VGG16
)
from project_config import *

def build_baseline_cnn():
    """
    Build a baseline CNN model from scratch
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Block 4
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model

def build_transfer_learning_model(base_model_name='MobileNetV2', trainable=False):
    """
    Build a transfer learning model
    
    Args:
        base_model_name: Name of pretrained model ('MobileNetV2', 'EfficientNetB0', 'ResNet50', 'VGG16')
        trainable: Whether to make base model trainable
        
    Returns:
        Compiled Keras model
    """
    # Select base model
    if base_model_name == 'MobileNetV2':
        base_model = MobileNetV2(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'EfficientNetB0':
        base_model = EfficientNetB0(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'ResNet50':
        base_model = ResNet50(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            include_top=False,
            weights='imagenet'
        )
    elif base_model_name == 'VGG16':
        base_model = VGG16(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Unknown base model: {base_model_name}")
    
    # Freeze/unfreeze base model
    base_model.trainable = trainable
    
    # Build model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    return model

def get_callbacks(model_name):
    """
    Get training callbacks
    
    Args:
        model_name: Name for saving checkpoints
        
    Returns:
        List of callbacks
    """
    callbacks = [
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(CHECKPOINT_PATH, f'{model_name}_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_DIR, model_name),
            histogram_freq=1
        )
    ]
    
    return callbacks

if __name__ == "__main__":
    # Test model building
    print("Building Baseline CNN...")
    baseline_model = build_baseline_cnn()
    baseline_model.summary()
    
    print("\nBuilding MobileNetV2 Transfer Learning Model...")
    mobilenet_model = build_transfer_learning_model('MobileNetV2')
    mobilenet_model.summary()
    
    print("\n✓ Models built successfully!")
