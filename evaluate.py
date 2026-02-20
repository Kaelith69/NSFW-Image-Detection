"""
evaluate.py
-----------
Evaluation script for the trained NSFW detection model.

Usage:
    python evaluate.py

The script loads the saved model, evaluates it on the validation and test
sets, and prints a full classification report together with a confusion
matrix.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

from preprocessing import validation_generator, test_generator

# ------------------------------------------------------------------
# Load model
# ------------------------------------------------------------------
MODEL_PATH = "nsfw_detector_model.keras"

model = load_model(MODEL_PATH)
print(f"Loaded model from: {MODEL_PATH}\n")

# ------------------------------------------------------------------
# Evaluate on validation set
# ------------------------------------------------------------------
val_loss, val_accuracy = model.evaluate(validation_generator, verbose=1)
print(f"\nValidation Loss:     {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%\n")

# ------------------------------------------------------------------
# Evaluate on test set
# ------------------------------------------------------------------
test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss:     {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")

# ------------------------------------------------------------------
# Classification report & confusion matrix (test set)
# ------------------------------------------------------------------
# Reset generator to the start to align predictions with true labels
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred = (y_pred_probs.ravel() >= 0.5).astype(int)
y_true = test_generator.classes

class_names = list(test_generator.class_indices.keys())  # e.g. ['nsfw', 'sfw']

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved to: confusion_matrix.png")
