import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def convert_to_grayscale_3channel(image):
    """
    Converts an RGB image to grayscale and then duplicates the channel 
    to make it 3-channel (required by EfficientNetB0).
    """
    if image.dtype != np.uint8:
        image_uint8 = image.astype(np.uint8)
    else:
        image_uint8 = image
        
    gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
    gray_3c = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return gray_3c.astype(image.dtype)

def get_data_generators(data_dir, batch_size=32, target_size=(224, 224)):
    """
    Sets up the data generators for training with augmentation and validation/testing.
    Assumes data_dir has 'train', 'val', and 'test' subdirectories with 'real' and 'fake' subfolders.
    """
    
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')
    
    # 1. Training generator with data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # 2. Validation/Test generators (no augmentation, only resizing)
    val_test_datagen = ImageDataGenerator()
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        print(f"Warning: Training directory {train_dir} not found.")
    if not os.path.exists(val_dir):
        print(f"Warning: Validation directory {val_dir} not found.")
    if not os.path.exists(test_dir):
        print(f"Warning: Test directory {test_dir} not found.")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    ) if os.path.exists(train_dir) else None
    
    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary'
    ) if os.path.exists(val_dir) else None
    
    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False # Important for evaluation metrics (y_true vs y_pred alignment)
    ) if os.path.exists(test_dir) else None
    
    return train_generator, val_generator, test_generator
