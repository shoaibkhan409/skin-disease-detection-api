from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image, ImageEnhance
import cv2
import os
import shutil
from huggingface_hub import hf_hub_download
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# ── Download model from Hugging Face if not present ───────────────────────────
KERAS_PATH   = 'best_skin_model.keras'
TFLITE_PATH  = 'skin_disease_model.tflite'
HF_REPO_ID   = 'shoaibb882/skin-disease-model'
HF_FILENAME  = 'best_skin_model.keras'

if not os.path.exists(KERAS_PATH):
    print("Downloading model from Hugging Face...")
    downloaded_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
    shutil.copy(downloaded_path, KERAS_PATH)
    print(f"Model downloaded successfully! Size: {os.path.getsize(KERAS_PATH) / (1024*1024):.1f} MB")

# ── Load model: Try best_skin_model.keras first, fallback to tflite ───────────
USE_KERAS = False
model_keras = None
interpreter = None
_input_details = None
_output_details = None

if os.path.exists(KERAS_PATH):
    try:
        model_keras = tf.keras.models.load_model(KERAS_PATH, compile=False)
        USE_KERAS = True
        print(f"[OK] Loaded Keras model: {KERAS_PATH}")
    except Exception as e:
        print(f"[WARN] Could not load Keras model: {e}")

if not USE_KERAS:
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()
    _input_details  = interpreter.get_input_details()
    _output_details = interpreter.get_output_details()
    print(f"[OK] Loaded TFLite model: {TFLITE_PATH}")

# ── Class names (must match training order) ───────────────────────────────────
CLASS_NAMES = ['Acne', 'Chickenpox', 'Dyshidrotic Eczema', 'Ringworm']

# ── Tuning knobs ──────────────────────────────────────────────────────────────
MIN_SKIN_RATIO           = 0.15   # minimum skin-pixel fraction
MIN_CONFIDENCE_THRESHOLD = 50.0   # minimum % confidence for a valid disease result

# ── Skin-pixel detector ───────────────────────────────────────────────────────
def calculate_skin_ratio(pil_img: Image.Image) -> float:
    """
    Returns fraction [0-1] of pixels that look like human skin using
    multi-color-space rules (YCrCb + HSV + RGB).
    """
    try:
        rgb  = np.array(pil_img.convert('RGB'))
        bgr  = cv2.cvtColor(rgb,  cv2.COLOR_RGB2BGR)
        ycc  = cv2.cvtColor(bgr,  cv2.COLOR_BGR2YCrCb)
        hsv  = cv2.cvtColor(bgr,  cv2.COLOR_BGR2HSV)

        # YCrCb range (works well for diverse skin tones)
        mask_ycc = cv2.inRange(ycc,
                               np.array([0,   133, 77],  dtype=np.uint8),
                               np.array([255, 173, 127], dtype=np.uint8))

        # HSV range – warm hues only
        mask_hsv = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,  20, 50],  dtype=np.uint8),
                             np.array([25, 255, 255], dtype=np.uint8)),
            cv2.inRange(hsv, np.array([160, 20, 50],  dtype=np.uint8),
                             np.array([180, 255, 255], dtype=np.uint8)),
        )

        # Classical RGB rule
        r, g, b = rgb[:, :, 0].astype(int), rgb[:, :, 1].astype(int), rgb[:, :, 2].astype(int)
        mask_rgb = (
            (r > 95) & (g > 40) & (b > 20) &
            ((np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) > 15) &
            (np.abs(r - g) > 15) & (r > g) & (r > b)
        )

        combined = ((cv2.bitwise_and(mask_ycc, mask_hsv) > 0) | mask_rgb)
        return float(np.mean(combined))
    except Exception as e:
        print(f"[skin_ratio] error: {e}")
        return 1.0  # fail-open

# ── Image pre-processor ───────────────────────────────────────────────────────
def preprocess_for_model(pil_img: Image.Image) -> np.ndarray:
    """
    Resize, lightly enhance contrast, normalise to [0, 1], add batch dim.
    """
    img = pil_img.convert('RGB').resize((224, 224), Image.LANCZOS)
    img = ImageEnhance.Contrast(img).enhance(1.15)
    img = ImageEnhance.Sharpness(img).enhance(1.10)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ── Inference functions ───────────────────────────────────────────────────────
def run_keras(img_array: np.ndarray) -> np.ndarray:
    preds = model_keras.predict(img_array, verbose=0)
    return preds[0]

def run_tflite(img_array: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(_input_details[0]['index'], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(_output_details[0]['index'])[0]

# ── TTA helper (Test-Time Augmentation with 5 variants for better accuracy) ───
def predict_with_tta(pil_img: Image.Image) -> np.ndarray:
    """
    Average predictions over multiple augmentations to reduce randomness
    and improve stability/accuracy.
    """
    variants = [
        pil_img,
        pil_img.transpose(Image.FLIP_LEFT_RIGHT),
        ImageEnhance.Brightness(pil_img).enhance(1.1),
        ImageEnhance.Brightness(pil_img).enhance(0.9),
        pil_img.rotate(5),
    ]
    preds = []
    for v in variants:
        arr = preprocess_for_model(v)
        if USE_KERAS:
            preds.append(run_keras(arr))
        else:
            preds.append(run_tflite(arr))
    return np.mean(preds, axis=0)

# ── /predict endpoint ─────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'success': False, 'error': 'No image file provided.'}), 400

        pil_img = Image.open(file.stream)

        # ── Step 1: Skin check ────────────────────────────────────────────────
        skin_ratio = calculate_skin_ratio(pil_img)
        skin_pct   = round(skin_ratio * 100, 1)

        if skin_ratio < MIN_SKIN_RATIO:
            return jsonify({
                'success':          True,
                'is_skin':          False,
                'is_valid_disease': False,
                'prediction':       'Not Found',
                'confidence':       0.0,
                'skin_percentage':  skin_pct,
                'message': (
                    'No human skin detected in this image. '
                    'Please upload a clear, close-up photo of the affected skin area.'
                ),
                'probabilities': {n: 0.0 for n in CLASS_NAMES},
            })

        # ── Step 2: TTA inference ─────────────────────────────────────────────
        probs      = predict_with_tta(pil_img)
        pred_idx   = int(np.argmax(probs))
        confidence = float(np.max(probs) * 100)

        probabilities = {CLASS_NAMES[i]: round(float(probs[i] * 100), 2)
                         for i in range(len(CLASS_NAMES))}

        print(f"[PREDICT] Skin: {skin_pct}%  |  Probs: {probabilities}")
        print(f"[PREDICT] Best: {CLASS_NAMES[pred_idx]}  |  Confidence: {confidence:.1f}%")

        # ── Step 3: Confidence check ──────────────────────────────────────────
        if confidence < MIN_CONFIDENCE_THRESHOLD:
            return jsonify({
                'success':          True,
                'is_skin':          True,
                'is_valid_disease': False,
                'prediction':       'Not Found',
                'confidence':       round(confidence, 1),
                'skin_percentage':  skin_pct,
                'message': (
                    f'Skin detected ({skin_pct}%) but no condition recognized with '
                    f'sufficient certainty ({confidence:.1f}% < {MIN_CONFIDENCE_THRESHOLD}%). '
                    'Please consult a dermatologist or provide a clearer, better-lit photo.'
                ),
                'probabilities': probabilities,
            })

        # ── Step 4: Valid result ──────────────────────────────────────────────
        return jsonify({
            'success':          True,
            'is_skin':          True,
            'is_valid_disease': True,
            'prediction':       CLASS_NAMES[pred_idx],
            'confidence':       round(confidence, 1),
            'skin_percentage':  skin_pct,
            'probabilities':    probabilities,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':       'healthy',
        'model':        KERAS_PATH if USE_KERAS else TFLITE_PATH,
        'model_type':   'Keras (.keras)' if USE_KERAS else 'TFLite (.tflite)',
        'classes':      CLASS_NAMES,
        'skin_thresh':  f'{MIN_SKIN_RATIO * 100:.0f}%',
        'conf_thresh':  f'{MIN_CONFIDENCE_THRESHOLD:.0f}%',
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("=" * 60)
    print("  DermaScan AI Server")
    print("=" * 60)
    print(f"  Model type  : {'Keras (best_skin_model.keras)' if USE_KERAS else 'TFLite (skin_disease_model.tflite)'}")
    print(f"  Classes     : {CLASS_NAMES}")
    print(f"  Skin thresh : {MIN_SKIN_RATIO * 100:.0f}%")
    print(f"  Conf thresh : {MIN_CONFIDENCE_THRESHOLD:.0f}%")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=False)

