"""
models.py

CNN model definitions for OCT-based nAMD prediction.

Includes:
1. Simple CNN (lightweight, quick test / baseline)
2. Baseline CNN for OCT biomarker classification (main model)
3. Transfer learning using MobileNetV2 (advanced, optional)

Usage:
Select model in config.py:
    model_name = create_simple_cnn
    model_name = create_cnn_for_oct_biomarker
    model_name = create_transfer_model
"""

from keras import layers, models
import tensorflow as tf

# ------------------------------
# Data augmentation helper
# ------------------------------
def build_augmentation_layer(flip, zoom, contrast, noise):
    """
    Returns a function applying the specified augmentations.

    - flip: 'horizontal', 'vertical', 'horizontal_and_vertical', 'none'
    - zoom: float (e.g., 0.2)
    - contrast: float (e.g., 0.3)
    - noise: float (e.g., 0.05)

    Note:
    Custom implementation allows consistent augmentation across multiple train/test splits,
    which standard Keras utilities cannot handle.
    
    """
    def augment(x):
        if flip == "vertical":
            x = layers.RandomFlip("vertical")(x)
        elif flip == "horizontal":
            x = layers.RandomFlip("horizontal")(x)
        elif flip == "horizontal_and_vertical":
            x = layers.RandomFlip("horizontal_and_vertical")(x)
        x = layers.RandomZoom(zoom)(x)
        x = layers.RandomContrast(contrast)(x)
        x = layers.GaussianNoise(noise)(x)
        return x
    return augment

# ------------------------------
# 1. Simple CNN
# ------------------------------
def create_simple_cnn(input_shape, num_classes, augmentation):
    """
    Lightweight CNN for quick tests or baseline experiments
    """
    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs)
    x = layers.Conv2D(32, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3,3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs)

# ------------------------------
# 2. Baseline CNN for OCT biomarker classification
# ------------------------------
def create_cnn_for_oct_biomarker(input_shape, num_classes, augmentation):
    """
    Main CNN model for OCT biomarker classification
    """
    inputs = layers.Input(shape=input_shape)
    x = augmentation(inputs)
    
    for filters in [32, 64, 128, 256]:
        x = layers.Conv2D(filters, (3,3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2,2))(x)
        x = layers.BatchNormalization()(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)

# ------------------------------
# 3. Transfer learning with MobileNetV2
# ------------------------------
_cached_base_model = None

def get_cached_base_model(input_shape):
    """
    Load MobileNetV2 base model and cache it to avoid reloading
    """
    global _cached_base_model
    if _cached_base_model is None:
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(input_shape[0], input_shape[1], 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        _cached_base_model = base_model
    return _cached_base_model

def create_transfer_model(input_shape, num_classes, augmentation, unfreeze_last_n=50):
    """
    Transfer learning model using MobileNetV2 as base.
    - input: grayscale OCT images (replicated to 3 channels)
    - unfreeze_last_n: fine-tune last n layers
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Concatenate()([inputs, inputs, inputs])  # replicate channels
    x = augmentation(x)
    
    base_model = get_cached_base_model(input_shape)
    base_model.trainable = True
    if unfreeze_last_n > 0:
        for layer in base_model.layers[:-unfreeze_last_n]:
            layer.trainable = False
    
    x = base_model(x, training=True)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.base_model = base_model
    return model