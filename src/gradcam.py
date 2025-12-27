import numpy as np
import cv2
import tensorflow as tf
from keras import layers

def make_gradcam_heatmap(img_array, model, pred_index=None):
    # Find the last convolutional layer in the model
    last_conv_layer = next(
        layer for layer in reversed(model.layers)
        if isinstance(layer, layers.Conv2D)
    )

    # Build a model that outputs feature maps from the last conv layer
    last_conv_model = tf.keras.Model(model.inputs, last_conv_layer.output) 

    # Define input for a small classifier on top of conv outputs
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    
    # Global average pooling converts feature maps into a single vector per channel
    x = layers.GlobalAveragePooling2D()(classifier_input)
    x = layers.Dense(64, activation='relu')(x) # Fully connected layer to combine features
    # Output layer with same number of classes as original model
    x = layers.Dense(model.output_shape[-1], activation='softmax')(x)
    
    classifier_model = tf.keras.Model(classifier_input, x) # Build classifier model

    # Record operations for gradient computation
    with tf.GradientTape() as tape:
        conv_outputs = last_conv_model(img_array, training=False) # Forward pass through conv layer
        
        tape.watch(conv_outputs) # Track conv outputs for gradients
        preds = classifier_model(conv_outputs) # Forward pass through classifier

        if pred_index is None:
            pred_index = tf.argmax(preds[0]) # If no target class is specified, take predicted class

        class_channel = preds[:, pred_index] # Select output score for target class

    grads = tape.gradient(class_channel, conv_outputs) # Compute gradients of class score w.r.t. conv outputs
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))# Average gradients over spatial dimensions
    
    conv_outputs = conv_outputs[0] # Get conv outputs for the first image in batch
    # Weight each feature map by its pooled gradient importance
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap) # Remove extra dimension

    # Keep only positive contributions and normalise
    heatmap = tf.maximum(heatmap, 0) / tf.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()

def save_and_display_gradcam(img, heatmap, alpha=0.4, output_path=None):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) # Resize heatmap to match original image size
    heatmap = np.uint8(255 * heatmap) # Convert heatmap to 0-255 scale

    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img_rgb = cv2.cvtColor((img*255).astype(np.uint8), cv2.COLOR_GRAY2RGB) if img.shape[-1]==1 else (img*255).astype(np.uint8)
    superimposed = cv2.addWeighted(img_rgb, 1-alpha, heatmap_color, alpha, 0) # Superimpose heatmap onto original image

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR))
