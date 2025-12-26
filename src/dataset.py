import os
import tensorflow as tf
from keras import layers

def list_image_filenames(directory):
    """
    List all image filenames in a directory 
    (only file names with .jpg, .png, .jpeg extensions).
    """
    return [
        f for root, _, files in os.walk(directory)
        for f in files if f.lower().endswith((".jpg",".png",".jpeg"))
    ]

def create_dataset_from_directory(directory, image_size=(450,450), batch_size=2, shuffle=True):
    """
    Create a TensorFlow dataset from directory with grayscale images and normalization.
    """
    dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels='inferred',
        label_mode='categorical',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=shuffle,
        color_mode='grayscale'
    )
    normalization_layer = layers.Rescaling(1./255)
    return dataset.map(lambda x, y: (normalization_layer(x), y))