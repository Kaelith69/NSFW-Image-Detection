"""
train.py
--------
Training script for the NSFW detection model.

Usage:
    python train.py

The script:
1. Loads data generators from preprocessing.py.
2. Builds the MobileNetV2-based model from model.py.
3. Trains with early stopping and learning-rate reduction callbacks.
4. Saves the best model checkpoint and the final model.
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks as keras_callbacks

from model import create_model
from preprocessing import train_generator, validation_generator

# ------------------------------------------------------------------
# Hyper-parameters
# ------------------------------------------------------------------
EPOCHS = 20
LEARNING_RATE = 1e-4
MODEL_SAVE_PATH = "nsfw_detector_model.keras"  # Use .keras format (recommended over .h5)

# ------------------------------------------------------------------
# Build model
# ------------------------------------------------------------------
model = create_model(learning_rate=LEARNING_RATE)
model.summary()

# ------------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------------
callbacks = [
    # Stop training if validation loss stops improving
    keras_callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
    ),
    # Reduce LR when validation loss plateaus
    keras_callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.2,
        patience=2,
        min_lr=1e-7,
    ),
    # Save best model during training
    keras_callbacks.ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor="val_loss",
        save_best_only=True,
    ),
]

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
)

print(f"\nModel saved to: {MODEL_SAVE_PATH}")

# ------------------------------------------------------------------
# Plot training history
# ------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history["accuracy"], label="Train accuracy")
axes[0].plot(history.history["val_accuracy"], label="Val accuracy")
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epoch")
axes[0].legend()

axes[1].plot(history.history["loss"], label="Train loss")
axes[1].plot(history.history["val_loss"], label="Val loss")
axes[1].set_title("Loss")
axes[1].set_xlabel("Epoch")
axes[1].legend()

plt.tight_layout()
plt.savefig("training_history.png")
print("Training history plot saved to: training_history.png")
