"""
preprocessing.py
----------------
Image preprocessing and data-generator setup for NSFW detection training.

Expected directory layout:
    data/
    ├── train/
    │   ├── nsfw/
    │   └── sfw/
    ├── validation/
    │   ├── nsfw/
    │   └── sfw/
    └── test/
        ├── nsfw/
        └── sfw/
"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
TRAIN_DIR = "data/train"
VALIDATION_DIR = "data/validation"
TEST_DIR = "data/test"

# Image dimensions expected by MobileNetV2
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ------------------------------------------------------------------
# Data generators
# ------------------------------------------------------------------

# Training generator with augmentation to reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

# Validation / test generators — only rescale, no augmentation
validation_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True,
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,
)
