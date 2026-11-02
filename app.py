
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import os
import shutil
import traceback
from huggingface_hub import hf_hub_download
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# --------------------------------------------------
# Model configuration
# --------------------------------------------------

KERAS_PATH = "best_skin_model.keras"
HF_REPO_ID = "shoaibb882/skin-disease-model"
HF_FILENAME = "best_skin_model.keras"

USE_KERAS = False
model_keras = None
MODEL_ERROR = None

# --------------------------------------------------
# Download and load Keras model
# --------------------------------------------------

try:
    if not os.path.exists(KERAS_PATH):
        print("Downloading model from Hugging Face...")

        downloaded_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME
        )

        shutil.copy(downloaded_path, KERAS_PATH)

        print(
            f"Model downloaded: "
            f"{os.path.getsize(KERAS_PATH) / (1024 * 1024):.1f} MB"
        )

    print("Loading Keras model...")

    model_keras = tf.keras.models.load_model(
        KERAS_PATH,
        compile=False
    )

    USE_KERAS = True

    print("[OK] Keras model loaded successfully.")

except Exception as e:
    MODEL_ERROR = str(e)

    print("[ERROR] Could not load Keras model:")
    traceback.print_exc()

    model_keras = None
    USE_KERAS = False

# --------------------------------------------------
# Class names
# Must match the training order
# --------------------------------------------------

CLASS_NAMES = [
    "Acne",
    "Chickenpox",
    "Dyshidrotic Eczema",
    "Ringworm"
]

MIN_SKIN_RATIO = 0.15
MIN_CONFIDENCE_THRESHOLD = 50.0

# --------------------------------------------------
# Skin detection
# --------------------------------------------------

def calculate_skin_ratio(pil_img: Image.Image) -> float:
    try:
        rgb = np.array(pil_img.convert("RGB"))

        bgr = cv2.cvtColor(
            rgb,
            cv2.COLOR_RGB2BGR
        )

        ycc = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2YCrCb
        )

        hsv = cv2.cvtColor(
            bgr,
            cv2.COLOR_BGR2HSV
        )

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

        r = rgb[:, :, 0].astype(int)
        g = rgb[:, :, 1].astype(int)
        b = rgb[:, :, 2].astype(int)

        mask_rgb = (
            (r > 95)
            & (g > 40)
            & (b > 20)
            & (
                np.maximum(
                    np.maximum(r, g), b
                )
                - np.minimum(
                    np.minimum(r, g), b
                ) > 15
            )
            & (np.abs(r - g) > 15)
            & (r > g)
            & (r > b)
        )

        combined = (
            (cv2.bitwise_and(mask_ycc, mask_hsv) > 0)
            | mask_rgb
        )

        return float(np.mean(combined))

    except Exception as e:
        print(f"[skin_ratio] error: {e}")
        return 1.0

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
    preds = model_keras.predict(
        img_array,
        verbose=0
    )

    return np.asarray(preds[0])

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
# Prediction endpoint
# --------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    if not USE_KERAS:
        return jsonify({
            "success": False,
            "error": "AI model is not loaded.",
            "details": MODEL_ERROR
        }), 503

    try:
        file = request.files.get("image")

        if not file:
            return jsonify({
                "success": False,
                "error": "No image file provided."
            }), 400

        pil_img = Image.open(file.stream).convert("RGB")

        # Step 1: Skin detection

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
                    "No human skin detected in this image. "
                    "Please upload a clear, close-up photo "
                    "of the affected skin area."
                ),
                "probabilities": {
                    name: 0.0 for name in CLASS_NAMES
                }
            })

        # Step 2: Model prediction

        probs = predict_with_tta(pil_img)

        if len(probs) != len(CLASS_NAMES):
            raise ValueError(
                f"Model returned {len(probs)} classes, "
                f"but {len(CLASS_NAMES)} class names are configured."
            )

        pred_idx = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100)

        probabilities = {
            CLASS_NAMES[i]: round(float(probs[i] * 100), 2)
            for i in range(len(CLASS_NAMES))
        }

        print(f"[PREDICT] Skin: {skin_pct}%")
        print(f"[PREDICT] Probabilities: {probabilities}")

        # Step 3: Confidence check

        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return jsonify({
                "success": True,
                "is_skin": True,
                "is_valid_disease": False,
                "prediction": "Not Found",
                "confidence": round(confidence, 1),
                "skin_percentage": skin_pct,
                "message": (
                    "Skin detected, but no condition recognized "
                    "with sufficient certainty. "
                    "Please consult a dermatologist."
                ),
                "probabilities": probabilities
            })

        # Step 4: Successful prediction

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
        "status": "healthy" if USE_KERAS else "model_error",
        "model": KERAS_PATH,
        "model_type": "Keras (.keras)",
        "model_loaded": USE_KERAS,
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
