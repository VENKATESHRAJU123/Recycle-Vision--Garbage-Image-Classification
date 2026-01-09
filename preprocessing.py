"""
Image preprocessing and data preparation functions
"""

import os
import shutil
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from project_config import *

def split_dataset(source_dir, dest_dir, test_size=0.2, val_size=0.1, random_state=42):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        source_dir: Path to raw data directory
        dest_dir: Path to processed data directory
        test_size: Proportion of test set
        val_size: Proportion of validation set from training data
        random_state: Random seed for reproducibility
    """
    print("Starting dataset split...")
    
    # Create directories
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(dest_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        for class_name in CLASS_NAMES:
            os.makedirs(os.path.join(split_dir, class_name), exist_ok=True)
    
    # Process each class
    for class_name in CLASS_NAMES:
        class_path = os.path.join(source_dir, class_name)
        
        # Get all image files
        images = [f for f in os.listdir(class_path) 
                  if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Split into train and test
        train_val_images, test_images = train_test_split(
            images, test_size=test_size, random_state=random_state
        )
        
        # Split train into train and validation
        train_images, val_images = train_test_split(
            train_val_images, test_size=val_size, random_state=random_state
        )
        
        # Copy files to respective directories
        for img in train_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(dest_dir, 'train', class_name, img)
            shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(dest_dir, 'validation', class_name, img)
            shutil.copy2(src, dst)
        
        for img in test_images:
            src = os.path.join(class_path, img)
            dst = os.path.join(dest_dir, 'test', class_name, img)
            shutil.copy2(src, dst)
        
        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val, {len(test_images)} test")
    
    print("✓ Dataset split completed!")

def create_data_generators():
    """
    Create data generators with augmentation for training and validation
    
    Returns:
        train_generator, val_generator, test_generator
    """
    # Training data generator with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        shear_range=SHEAR_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode=FILL_MODE
    )
    
    # Validation and test data generators (only rescaling)
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'validation'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(PROCESSED_DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✓ Data generators created!")
    print(f"  Training samples: {train_generator.samples}")
    print(f"  Validation samples: {val_generator.samples}")
    print(f"  Test samples: {test_generator.samples}")
    
    return train_generator, val_generator, test_generator

def preprocess_image(image_path):
    """
    Preprocess a single image for prediction
    
    Args:
        image_path: Path to image file
        
    Returns:
        Preprocessed image array
    """
    img = Image.open(image_path)
    img = img.resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

if __name__ == "__main__":
    # Split dataset
    split_dataset(RAW_DATA_DIR, PROCESSED_DATA_DIR)
    
    # Create data generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    print("\n✓ Preprocessing completed successfully!")
