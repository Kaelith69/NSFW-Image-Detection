<div align="center">

<!-- Hero SVG Banner -->
<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 900 220" width="900" height="220">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0f0c29"/>
      <stop offset="50%" style="stop-color:#302b63"/>
      <stop offset="100%" style="stop-color:#24243e"/>
    </linearGradient>
    <linearGradient id="accent" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#f953c6"/>
      <stop offset="100%" style="stop-color:#b91d73"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="coloredBlur"/>
      <feMerge><feMergeNode in="coloredBlur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <!-- Background -->
  <rect width="900" height="220" fill="url(#bg)" rx="16"/>
  <!-- Decorative circuit lines -->
  <g stroke="#f953c6" stroke-width="0.8" opacity="0.18">
    <line x1="0" y1="40" x2="900" y2="40"/>
    <line x1="0" y1="180" x2="900" y2="180"/>
    <circle cx="120" cy="40" r="4" fill="#f953c6"/>
    <circle cx="780" cy="40" r="4" fill="#f953c6"/>
    <circle cx="120" cy="180" r="4" fill="#f953c6"/>
    <circle cx="780" cy="180" r="4" fill="#f953c6"/>
    <line x1="120" y1="40" x2="120" y2="180"/>
    <line x1="780" y1="40" x2="780" y2="180"/>
  </g>
  <!-- Shield icon -->
  <g transform="translate(60, 65)" filter="url(#glow)">
    <path d="M45 0 L90 18 L90 54 Q90 85 45 100 Q0 85 0 54 L0 18 Z"
          fill="url(#accent)" opacity="0.9"/>
    <path d="M45 22 L60 30 L60 50 Q60 64 45 72 Q30 64 30 50 L30 30 Z"
          fill="white" opacity="0.9"/>
    <text x="45" y="55" text-anchor="middle" font-size="22" font-weight="bold"
          fill="#0f0c29" font-family="monospace">🛡</text>
  </g>
  <!-- Main title -->
  <text x="185" y="100" font-family="'Courier New', monospace" font-size="38"
        font-weight="bold" fill="white" filter="url(#glow)">NSFW Image</text>
  <text x="185" y="148" font-family="'Courier New', monospace" font-size="38"
        font-weight="bold" fill="url(#accent)" filter="url(#glow)">Detection</text>
  <!-- Tagline -->
  <text x="185" y="178" font-family="Arial, sans-serif" font-size="14"
        fill="#aaaacc" letter-spacing="2">AI-powered · MobileNetV2 · Flask REST API</text>
  <!-- Version pill -->
  <rect x="680" y="85" width="160" height="32" rx="16" fill="url(#accent)" opacity="0.85"/>
  <text x="760" y="106" text-anchor="middle" font-family="monospace" font-size="13"
        fill="white" font-weight="bold">v2.0 · Python 3.x</text>
</svg>

---

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white&style=flat-square)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12%2B-FF6F00?logo=tensorflow&logoColor=white&style=flat-square)](https://tensorflow.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white&style=flat-square)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Transformers-FFD21E?style=flat-square)](https://huggingface.co)
[![Flask](https://img.shields.io/badge/Flask-2.3%2B-000000?logo=flask&logoColor=white&style=flat-square)](https://flask.palletsprojects.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Version](https://img.shields.io/badge/version-2.0.0-8b5cf6?style=flat-square)]()

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Setup & Installation](#-setup--installation)
- [Data Preparation](#-data-preparation)
- [Training the Model](#-training-the-model)
- [Evaluating the Model](#-evaluating-the-model)
- [Running the API](#-running-the-api)
- [API Reference](#-api-reference)
- [Deployment](#-deployment)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

**NSFW Image Detection** is a dual-path machine learning system that classifies images as **NSFW** (Not Safe For Work) or **SFW** (Safe For Work).

| Path | Use Case | Model |
|------|----------|-------|
| **Transfer Learning** | Custom datasets, fine-tuning | MobileNetV2 + TensorFlow |
| **Zero-Shot API** | Quick deployment, no training data needed | `Falconsai/nsfw_image_detection` via Hugging Face |

Both paths are served through the same **Flask REST API**, so you can swap backends without changing your client code.

---

## 🏗 Architecture

```
                    ┌─────────────────────────────────────┐
                    │           Flask REST API             │
                    │           (app.py :5000)             │
                    └────────────────┬────────────────────┘
                                     │ POST /predict
                         ┌───────────▼───────────┐
                         │   Image Validation     │
                         │   (type, size, format) │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  Hugging Face Model    │
                         │  Falconsai/nsfw_image  │
                         │  _detection (PyTorch)  │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │  Softmax → id2label    │
                         │  {"nsfw": 0.97, ...}   │
                         └───────────┬───────────┘
                                     │
                         ┌───────────▼───────────┐
                         │   JSON Response        │
                         │   prediction + scores  │
                         └───────────────────────┘

  ── Custom Training Path ──────────────────────────────────────────
  raw images → preprocessing.py → model.py (MobileNetV2) → train.py
                                                        ↓
                                              nsfw_detector_model.keras
                                                        ↓
                                                  evaluate.py
```

---

## ✨ Features

- 🔒 **Secure file handling** — `werkzeug.secure_filename`, extension allow-list, 16 MB upload cap, automatic temp-file cleanup
- 🏷 **Label-safe predictions** — uses `model.config.id2label` instead of hardcoded indices
- 📉 **Smart training callbacks** — EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- 📊 **Rich evaluation** — classification report + confusion matrix PNG output
- 🌡 **Health endpoint** — `GET /health` for load-balancer checks
- 🐛 **Debug-mode gating** — `FLASK_DEBUG=1` env var (never enabled by default)
- 📦 **Modern save format** — `.keras` (replaces deprecated `.h5`)

---

## 📁 Project Structure

```
nsfw-image-detection/
│
├── 📄 app.py              # Flask REST API — /predict & /health endpoints
├── 📄 model.py            # MobileNetV2 transfer-learning model factory
├── 📄 preprocessing.py    # ImageDataGenerator setup for train/val/test
├── 📄 train.py            # Training loop with callbacks + history plot
├── 📄 evaluate.py         # Metrics, classification report, confusion matrix
├── 📄 requirements.txt    # Pinned Python dependencies
├── 📄 README.md           # You are here
│
├── 📂 data/               # (gitignored) Your image dataset
│   ├── 📂 train/
│   │   ├── 📂 nsfw/       # Training NSFW images
│   │   └── 📂 sfw/        # Training SFW images
│   ├── 📂 validation/
│   │   ├── 📂 nsfw/
│   │   └── 📂 sfw/
│   └── 📂 test/
│       ├── 📂 nsfw/
│       └── 📂 sfw/
│
├── 📂 uploads/            # (auto-created, gitignored) Temp API uploads
├── 🖼  training_history.png # Generated after training
└── 🖼  confusion_matrix.png  # Generated after evaluation
```

---

## 📦 Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.9+ |
| TensorFlow | ≥ 2.12 |
| PyTorch | ≥ 2.0 |
| HuggingFace Transformers | ≥ 4.35 |
| Flask | ≥ 2.3 |
| Pillow | ≥ 9.5 |
| scikit-learn | ≥ 1.2 |

---

## 🚀 Setup & Installation

### 1 — Clone the repository

```bash
git clone https://github.com/Kaelith69/NSFW-Image-Detection.git
cd NSFW-Image-Detection
```

### 2 — Create and activate a virtual environment

```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🗂 Data Preparation

### Create the folder structure

```bash
# macOS / Linux / Windows (Git Bash)
mkdir -p data/train/nsfw data/train/sfw \
         data/validation/nsfw data/validation/sfw \
         data/test/nsfw data/test/sfw
```

Place your images in the appropriate sub-folders:

```
data/
├── train/
│   ├── nsfw/   ← NSFW training images
│   └── sfw/    ← SFW training images
├── validation/
│   ├── nsfw/   ← NSFW validation images
│   └── sfw/    ← SFW validation images
└── test/
    ├── nsfw/
    └── sfw/
```

> **Tip:** Aim for a balanced dataset (roughly equal NSFW and SFW counts). A minimum of ~500 images per class is recommended.

---

## 🏋 Training the Model

> 🎉 ![Training in progress](https://media.giphy.com/media/du3J3cXyzhj75IOgvA/giphy.gif)

```bash
python train.py
```

What this does:
1. Loads augmented data generators from `preprocessing.py`
2. Builds the MobileNetV2 transfer-learning model from `model.py`
3. Trains for up to **20 epochs** with EarlyStopping (patience = 4)
4. Saves the best checkpoint as **`nsfw_detector_model.keras`**
5. Saves a training history plot as **`training_history.png`**

**To fine-tune the base model** (after initial training converges), edit `train.py` and update `create_model()`:

```python
# Unfreeze layers above index 100 for fine-tuning
model = create_model(learning_rate=1e-5, fine_tune_at=100)
```

---

## 📊 Evaluating the Model

```bash
python evaluate.py
```

Output:
- Validation and test accuracy/loss printed to the console
- Full classification report (precision, recall, F1)
- Confusion matrix saved as **`confusion_matrix.png`**

---

## 🌐 Running the API

### Start the server

```bash
python app.py
```

The API starts on **http://localhost:5000**.

### Enable debug mode (development only)

```bash
FLASK_DEBUG=1 python app.py
```

> ⚠️ **Never set `FLASK_DEBUG=1` in production.**

---

## 📡 API Reference

### `GET /health`

Liveness check for load balancers and monitoring.

```bash
curl http://localhost:5000/health
```

```json
{ "status": "ok" }
```

---

### `POST /predict`

Classify an image as NSFW or SFW.

**Request**

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | Image file (jpg, jpeg, png, gif, bmp, webp) — max 16 MB |

```bash
curl -X POST http://localhost:5000/predict \
     -F "image=@/path/to/your/photo.jpg"
```

**Success Response (200)**

```json
{
  "prediction": "normal",
  "confidence": 0.9831,
  "scores": {
    "normal": 0.9831,
    "nsfw": 0.0169
  }
}
```

**Error Responses**

| Status | Condition |
|--------|-----------|
| 400 | Missing `image` field or empty filename |
| 415 | Unsupported file type |
| 422 | File is not a valid image |

---

### Quick Python client example

```python
import requests

with open("photo.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:5000/predict",
        files={"image": f},
    )

data = response.json()
print(f"Prediction : {data['prediction']}")
print(f"Confidence : {data['confidence'] * 100:.1f}%")
print(f"All scores : {data['scores']}")
```

---

## 🚢 Deployment

### Docker (recommended)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "app:app"]
```

```bash
docker build -t nsfw-detector .
docker run -p 5000:5000 nsfw-detector
```

### Heroku

Follow the [Heroku Python deployment guide](https://devcenter.heroku.com/articles/getting-started-with-python).

### AWS Elastic Beanstalk

Follow the [AWS EB Flask guide](https://docs.aws.amazon.com/elasticbeanstalk/latest/dg/create-deploy-python-flask.html).

---

## 🙏 Acknowledgments

- [TensorFlow](https://tensorflow.org) — training infrastructure and MobileNetV2 weights
- [Hugging Face](https://huggingface.co) — Transformers library and the [`Falconsai/nsfw_image_detection`](https://huggingface.co/Falconsai/nsfw_image_detection) model
- [MobileNetV2](https://arxiv.org/abs/1801.04381) (Sandler et al., 2018) — efficient CNN backbone
- [Flask](https://flask.palletsprojects.com) — lightweight WSGI web framework

---

<div align="center">

*Made with ☕ and questionable life choices.*

**Why did the neural network break up with the dataset?**
*Because it had too many issues and never validated its inputs.* 🥁

</div>
