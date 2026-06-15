import os
import tensorflow as tf
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# Google Colab Setup
# ==========================================
print("Setting up Colab AI Pipeline...")
DATA_DIR = '/content/dataset' 
MODEL_DIR = '/content/model'
SPLIT_DIR = '/content/dataset_split'

os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_dataset():
    train_dir = os.path.join(DATA_DIR, 'train')
    if os.path.exists(train_dir): return DATA_DIR
    au_dir = os.path.join(DATA_DIR, 'Au')
    tp_dir = os.path.join(DATA_DIR, 'Tp')
    
    if os.path.exists(au_dir) and os.path.exists(tp_dir):
        print("⚠️ Splitting dataset into train/val automatically...")
        try: import splitfolders
        except ImportError:
            os.system('pip install split-folders')
            import splitfolders
            
        splitfolders.ratio(DATA_DIR, output=SPLIT_DIR, seed=42, ratio=(0.8, 0.2))
        return SPLIT_DIR
    return DATA_DIR

WORKING_DATA_DIR = prepare_dataset()

# ==========================================
# 1. Enhanced Data Generators
# ==========================================
def get_data_generators():
    train_dir = os.path.join(WORKING_DATA_DIR, 'train')
    val_dir = os.path.join(WORKING_DATA_DIR, 'val')
    
    train_datagen = ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.15,
        brightness_range=[0.8, 1.2],
        fill_mode='reflect' 
    )
    val_datagen = ImageDataGenerator()
    
    train_gen = train_datagen.flow_from_directory(
        train_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
    ) if os.path.exists(train_dir) else None
    
    val_gen = val_datagen.flow_from_directory(
        val_dir, target_size=(224, 224), batch_size=32, class_mode='binary'
    ) if os.path.exists(val_dir) else None
    
    return train_gen, val_gen

# ==========================================
# 2. EfficientNetB0 Profile
# ==========================================
def build_robust_model():
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False 
    
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    feature_layer = Dense(256, activation='relu')(x)
    x = Dropout(0.6)(feature_layer) 
    
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

# ==========================================
# 3. Execution Pipeline
# ==========================================
def run_training():
    train_gen, val_gen = get_data_generators()
    if not train_gen: return

    classes = train_gen.classes
    class_weights = compute_class_weight('balanced', classes=np.unique(classes), y=classes)
    cw_dict = dict(enumerate(class_weights))

    model = build_robust_model()

    print("\n🚀 Phase 1: Training Head Layers...")
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    
    cb_p1 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    model.fit(train_gen, validation_data=val_gen, epochs=15, callbacks=[cb_p1], class_weight=cw_dict)

    print("\n🔬 Phase 2: Gentle Base Unfreezing (Top 40 Layers)...")
    # 🎯 ACCURACY FIX: Unfreezing 40 layers instead of 20 gives the model more room to learn the specific CASIA2 textures
    for layer in model.layers[1].layers[-40:]:  
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = True

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss='binary_crossentropy', metrics=['accuracy'])
                  
    cb_p2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7)
    
    model.fit(train_gen, validation_data=val_gen, epochs=30, callbacks=[cb_p2, reduce_lr], class_weight=cw_dict)

    model.save(os.path.join(MODEL_DIR, 'colab_forgery_model.h5'))
    print("\n🎉 SUCCESS! Training completed.")

if __name__ == "__main__":
    run_training()
