"""
model.py
--------
Defines the MobileNetV2-based transfer-learning model for NSFW detection.

The model freezes the base MobileNetV2 weights so that only the custom
classification head is trained during initial training. A fine-tuning
step can optionally unfreeze the top layers of the base model.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)


def create_model(learning_rate: float = 1e-4, fine_tune_at: int = 0) -> tf.keras.Model:
    """Build and compile the NSFW detection model.

    Args:
        learning_rate: Initial learning rate for the Adam optimiser.
        fine_tune_at: Index of the MobileNetV2 layer from which to unfreeze
            weights for fine-tuning. Pass 0 (default) to keep the whole base
            model frozen.

    Returns:
        A compiled ``tf.keras.Model`` ready for training.
    """
    # Load MobileNetV2 pre-trained on ImageNet, without the top classifier
    base_model = MobileNetV2(
        input_shape=IMG_SHAPE,
        include_top=False,
        weights="imagenet",
    )

    # Freeze the base model (feature extraction only)
    base_model.trainable = False

    # Optionally unfreeze layers above `fine_tune_at` for fine-tuning
    if fine_tune_at > 0:
        base_model.trainable = True
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False

    # Build the classification head
    inputs = tf.keras.Input(shape=IMG_SHAPE)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    # Single sigmoid output: 0 → SFW, 1 → NSFW
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs, name="nsfw_detector")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    return model
