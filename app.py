"""
app.py
------
Flask REST API for NSFW image detection using the Hugging Face
Falconsai/nsfw_image_detection model.

Endpoints
---------
POST /predict
    Accepts a multipart/form-data request with an ``image`` field.
    Returns JSON with the predicted label and confidence scores.

GET /health
    Health-check endpoint.

Usage
-----
    python app.py

Environment variables
---------------------
FLASK_DEBUG   Set to "1" to enable debug mode (never use in production).
UPLOAD_FOLDER Path for temporary image storage (default: ./uploads).
"""

import os

import torch
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from transformers import AutoImageProcessor, AutoModelForImageClassification
from werkzeug.utils import secure_filename

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "./uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "gif", "bmp", "webp"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------------------------------------------
# Load model once at startup (not on every request)
# ------------------------------------------------------------------
MODEL_NAME = "Falconsai/nsfw_image_detection"
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
hf_model = AutoModelForImageClassification.from_pretrained(MODEL_NAME)
hf_model.eval()  # Set to inference mode

# ------------------------------------------------------------------
# Flask app
# ------------------------------------------------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


def _allowed_file(filename: str) -> bool:
    """Return True if *filename* has an allowed image extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok"}), 200


@app.route("/predict", methods=["POST"])
def predict():
    """Classify an uploaded image as NSFW or SFW.

    Request
    -------
    Content-Type: multipart/form-data
    Field: image  — the image file to classify

    Response (200)
    --------------
    {
        "prediction": "nsfw" | "normal",
        "confidence": 0.98,
        "scores": {"nsfw": 0.98, "normal": 0.02}
    }
    """
    # --- Validate request ---
    if "image" not in request.files:
        return jsonify({"error": "No 'image' field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not _allowed_file(file.filename):
        return jsonify(
            {
                "error": (
                    f"Unsupported file type. Allowed types: "
                    f"{', '.join(sorted(ALLOWED_EXTENSIONS))}"
                )
            }
        ), 415

    # --- Save file safely ---
    filename = secure_filename(file.filename)
    img_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(img_path)

    try:
        # --- Open and validate image ---
        try:
            image = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            return jsonify({"error": "Uploaded file is not a valid image"}), 422

        # --- Run inference ---
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = hf_model(**inputs)

        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)[0]

        # Build a human-readable scores dict using the model's label mapping
        id2label = hf_model.config.id2label
        scores = {id2label[i]: round(probs[i].item(), 4) for i in range(len(probs))}

        predicted_id = int(logits.argmax(dim=1).item())
        predicted_label = id2label[predicted_id]
        confidence = round(probs[predicted_id].item(), 4)

        return jsonify(
            {
                "prediction": predicted_label,
                "confidence": confidence,
                "scores": scores,
            }
        ), 200

    finally:
        # Always clean up the temporary upload
        if os.path.exists(img_path):
            os.remove(img_path)


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, host="0.0.0.0", port=5000)
