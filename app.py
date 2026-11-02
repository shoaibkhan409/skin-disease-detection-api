
from flask import Flask, request, jsonify
from flask_cors import CORS

import os
import traceback
import numpy as np
import cv2
import tensorflow as tf

from PIL import Image, ImageEnhance
from huggingface_hub import hf_hub_download


# --------------------------------------------------
# Flask configuration
# --------------------------------------------------

app = Flask(__name__)
CORS(app)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

HF_REPO_ID = "shoaibb882/skin-disease-model"
HF_FILENAME = "best_skin_model.keras"

CLASS_NAMES = [
    "Acne",
    "Chickenpox",
    "Dyshidrotic Eczema",
    "Ringworm"
]

MIN_SKIN_RATIO = 0.15
MIN_CONFIDENCE_THRESHOLD = 50.0

model_keras = None
MODEL_ERROR = None
MODEL_PATH = None


# --------------------------------------------------
# Download and load the model
# --------------------------------------------------

def load_skin_model():
    global model_keras, MODEL_ERROR, MODEL_PATH

    try:
        print("Downloading model from Hugging Face...")

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME
        )

        MODEL_PATH = downloaded_path

        print(
            f"Model downloaded: "
            f"{os.path.getsize(downloaded_path) / (1024 * 1024):.1f} MB"
        )

        print("Loading Keras model...")

        loaded_model = tf.keras.models.load_model(
            downloaded_path,
            compile=False,
            safe_mode=True
        )

        # Validate model output
        if len(loaded_model.output_shape) != 2:
            raise ValueError(
                "Model must return a batch of class probabilities."
            )

        output_classes = loaded_model.output_shape[-1]

        if output_classes != len(CLASS_NAMES):
            raise ValueError(
                f"Model has {output_classes} output classes, "
                f"but {len(CLASS_NAMES)} class names are configured."
            )

        model_keras = loaded_model
        MODEL_ERROR = None

        print("[OK] Keras model loaded successfully.")
        print(f"Input shape: {model_keras.input_shape}")
        print(f"Output shape: {model_keras.output_shape}")

    except Exception as e:
        model_keras = None
        MODEL_ERROR = str(e)

        print("[ERROR] Could not load Keras model:")
        traceback.print_exc()


load_skin_model()


# --------------------------------------------------
# Skin detection
# --------------------------------------------------

def calculate_skin_ratio(pil_img: Image.Image) -> float:

    rgb = np.array(pil_img.convert("RGB"))

    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    mask_ycc = cv2.inRange(
        ycc,
        np.array([0, 133, 77], dtype=np.uint8),
        np.array([255, 173, 127], dtype=np.uint8)
    )

    mask_hsv = cv2.bitwise_or(
        cv2.inRange(
            hsv,
            np.array([0, 20, 50], dtype=np.uint8),
            np.array([25, 255, 255], dtype=np.uint8)
        ),
        cv2.inRange(
            hsv,
            np.array([160, 20, 50], dtype=np.uint8),
            np.array([180, 255, 255], dtype=np.uint8)
        )
    )

    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    mask_rgb = (
        (r > 95)
        & (g > 40)
        & (b > 20)
        & ((np.maximum(np.maximum(r, g), b)
            - np.minimum(np.minimum(r, g), b)) > 15)
        & (np.abs(r - g) > 15)
        & (r > g)
        & (r > b)
    )

    combined = (
        (cv2.bitwise_and(mask_ycc, mask_hsv) > 0)
        | mask_rgb
    )

    return float(np.mean(combined))


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def preprocess_for_model(pil_img: Image.Image) -> np.ndarray:

    img = pil_img.convert("RGB").resize(
        (224, 224),
        Image.Resampling.LANCZOS
    )

    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.10)

    arr = np.array(img, dtype=np.float32) / 255.0

    return np.expand_dims(arr, axis=0)


# --------------------------------------------------
# Keras inference
# --------------------------------------------------

def run_keras(img_array: np.ndarray) -> np.ndarray:

    if model_keras is None:
        raise RuntimeError("AI model is not loaded.")

    preds = model_keras.predict(
        img_array,
        verbose=0
    )

    probs = np.asarray(preds[0], dtype=np.float32)

    if not np.all(np.isfinite(probs)):
        raise ValueError("Model returned invalid predictions.")

    if np.any(probs < 0) or np.any(probs > 1):
        probs = tf.nn.softmax(probs).numpy()

    return probs


# --------------------------------------------------
# Test-time augmentation
# --------------------------------------------------

def predict_with_tta(pil_img: Image.Image) -> np.ndarray:

    variants = [
        pil_img,
        pil_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(pil_img).enhance(1.1),
        ImageEnhance.Brightness(pil_img).enhance(0.9),
        pil_img.rotate(5)
    ]

    preds = []

    for variant in variants:
        arr = preprocess_for_model(variant)
        preds.append(run_keras(arr))

    return np.mean(preds, axis=0)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Skin Disease Detection API",
        "status": "online",
        "model_loaded": model_keras is not None,
        "health_endpoint": "/health",
        "prediction_endpoint": "/predict"
    })


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if model_keras is None:
        return jsonify({
            "success": False,
            "error": "AI model is not loaded.",
            "details": MODEL_ERROR
        }), 503

    try:
        file = request.files.get("image")

        if file is None:
            return jsonify({
                "success": False,
                "error": "No image file provided."
            }), 400

        pil_img = Image.open(file.stream).convert("RGB")

        skin_ratio = calculate_skin_ratio(pil_img)
        skin_pct = round(skin_ratio * 100, 1)

        if skin_ratio < MIN_SKIN_RATIO:
            return jsonify({
                "success": True,
                "is_skin": False,
                "is_valid_disease": False,
                "prediction": "Not Found",
                "confidence": 0.0,
                "skin_percentage": skin_pct,
                "message": (
                    "No human skin detected. "
                    "Please upload a clear close-up photo "
                    "of the affected skin area."
                ),
                "probabilities": {
                    name: 0.0 for name in CLASS_NAMES
                }
            })

        probs = predict_with_tta(pil_img)

        if len(probs) != len(CLASS_NAMES):
            raise ValueError(
                f"Model returned {len(probs)} classes, "
                f"expected {len(CLASS_NAMES)}."
            )

        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100)

        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i] * 100), 2)
            for i in range(len(CLASS_NAMES))
        }

        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return jsonify({
                "success": True,
                "is_skin": True,
                "is_valid_disease": False,
                "prediction": "Not Found",
                "confidence": round(confidence, 1),
                "skin_percentage": skin_pct,
                "message": (
                    "Skin detected, but the model could not "
                    "recognize a condition with sufficient certainty. "
                    "Please consult a dermatologist."
                ),
                "probabilities": probabilities
            })

        return jsonify({
            "success": True,
            "is_skin": True,
            "is_valid_disease": True,
            "prediction": CLASS_NAMES[pred_idx],
            "confidence": round(confidence, 1),
            "skin_percentage": skin_pct,
            "probabilities": probabilities
        })

    except Exception as e:
        traceback.print_exc()

        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy" if model_keras is not None else "model_error",
        "model": HF_FILENAME,
        "model_loaded": model_keras is not None,
        "model_error": MODEL_ERROR,
        "classes": CLASS_NAMES,
        "skin_thresh": f"{MIN_SKIN_RATIO * 100:.0f}%",
        "conf_thresh": f"{MIN_CONFIDENCE_THRESHOLD:.0f}%"
    })


# --------------------------------------------------
# Start server
# --------------------------------------------------

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
