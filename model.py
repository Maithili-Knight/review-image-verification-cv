import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

def build_model(input_shape=(224, 224, 3)):
    """
    Builds the image verification model based on EfficientNetB0.
    """
    # Load the base model with pre-trained ImageNet weights
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze the base model to prevent weights from being updated during initial training
    base_model.trainable = False

    # Create the model head
    inputs = tf.keras.Input(shape=input_shape)
    
    # EfficientNetB0 includes preprocessing inside the model in tf.keras
    x = base_model(inputs, training=False)
    
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs, outputs)
    
    return model

if __name__ == '__main__':
    model = build_model()
    model.summary()
