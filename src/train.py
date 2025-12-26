import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from dataset import create_dataset_from_directory, list_image_filenames
from gradcam import make_gradcam_heatmap, save_and_display_gradcam

class TeeLogger:
    """
    Logger that writes to both terminal and a log file.
    """
    def __init__(self, file_path):
        self.terminal = sys.__stdout__  # always original stdout
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()
        sys.stdout = self.terminal

def run_split(split_name, base_dir, input_shape, augmentation_cache, augmentation_configs,
    model_func, save_dir, condition, batch_size=2, epochs=50, patience=7):
    """
    Train and evaluate a CNN model for a single data split.
    Logs, learning curves, TP/FP analysis, and Grad-CAM are saved.
    """
    # Prepare output directories
    log_dir = f"{save_dir}/logs/{condition}/"
    curve_dir = f"{save_dir}/learning_curves/{condition}/"
    gradcam_dir = f"{save_dir}/grad_cam/{condition}/"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(gradcam_dir, exist_ok=True)

    split_log_path = os.path.join(log_dir, f"cnn_{split_name}.txt")

    logger = TeeLogger(split_log_path)
    sys.stdout = logger

    try:
        # Load data
        train_dir = os.path.join(base_dir, split_name, "train")
        test_dir = os.path.join(base_dir, split_name, "test")

        train_filenames = list_image_filenames(train_dir)
        test_filenames = list_image_filenames(test_dir)

        print(f"[{split_name}] Train: {len(train_filenames)}, Test: {len(test_filenames)}")

        best_acc = 0.0
        best_model = None
        best_labels = []
        best_preds = []
        best_history = None

        # Train with different augmentations
        for config in augmentation_configs:
            augmentation_layer = augmentation_cache[config]

            train_dataset = create_dataset_from_directory(
                train_dir,
                image_size=input_shape[:2],
                batch_size=batch_size,
                shuffle=True,
            )
            test_dataset = create_dataset_from_directory(
                test_dir,
                image_size=input_shape[:2],
                batch_size=batch_size,
                shuffle=False,
            )

            num_classes = train_dataset.element_spec[1].shape[-1]

            model = model_func(
                input_shape=input_shape,
                num_classes=num_classes,
                augmentation=augmentation_layer,
            )

            model.compile(
                optimizer="adam",
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
            )

            history = model.fit(
                train_dataset,
                validation_data=test_dataset,
                epochs=epochs,
                verbose=0,
                callbacks=[early_stop],
            )

            _, test_acc = model.evaluate(test_dataset, verbose=0)

            if test_acc > best_acc:
                best_acc = test_acc
                best_config = config
                best_model = model
                best_history = history

                y_true, y_pred = [], []
                for images, labels in test_dataset:
                    preds = model.predict(images, verbose=0)
                    y_true.extend(np.argmax(labels.numpy(), axis=1))
                    y_pred.extend(np.argmax(preds, axis=1))

                best_labels = y_true
                best_preds = y_pred

        # Plot learning curves
        plt.figure(figsize=(8, 6))
        plt.plot(best_history.history["accuracy"], label="Train Accuracy")
        plt.plot(best_history.history["val_accuracy"], label="Validation Accuracy")
        plt.plot(best_history.history["loss"], label="Train Loss")
        plt.plot(best_history.history["val_loss"], label="Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(curve_dir, f"learning_curve_{split_name}.png"))
        plt.close()

        # Print F1-score for best model
        f1 = f1_score(best_labels, best_preds)
        print(f"F1-score: {f1:.4f}")

        # TP / FP / FN / TN analysis
        print("\nImage-level classification results:")
        for idx, (t, p) in enumerate(zip(best_labels, best_preds)):
            filename = test_filenames[idx]
            if t == 1 and p == 1:
                print(f"TP: {filename}")
            elif t == 0 and p == 1:
                print(f"FP: {filename}")
            elif t == 1 and p == 0:
                print(f"FN: {filename}")
            elif t == 0 and p == 0:
                print(f"TN: {filename}")

        # Grad-CAM visualization
        for batch_idx, (images, labels) in enumerate(test_dataset):
            for j in range(images.shape[0]):
                img = images[j : j + 1]
                label = np.argmax(labels[j].numpy())
                pred = np.argmax(best_model.predict(img, verbose=0))

                heatmap = make_gradcam_heatmap(
                    img, best_model, pred_index=pred
                )

                save_path = os.path.join(
                    gradcam_dir,
                    f"gradcam_{split_name}_{batch_idx}_{j}.png",
                )

                save_and_display_gradcam(
                    img[0].numpy(),
                    heatmap,
                    output_path=save_path,
                )

    finally:
        logger.close() # restore stdout and close logger