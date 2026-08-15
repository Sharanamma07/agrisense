"""
AgriSense — Flask app.

Serves an upload UI and a /predict API endpoint. Loads the trained model
once at startup and runs inference on uploaded leaf images.
"""

import json
import os
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model

MODEL_DIR = "models"
UPLOAD_DIR = os.path.join("static", "uploads")
IMG_SIZE = (224, 224)

app = Flask(__name__)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Load model + class names + treatment lookup once at startup ---
model = None
class_names = []
disease_info = {}

try:
    model = load_model(os.path.join(MODEL_DIR, "agrisense_model.keras"))
    with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
        class_names = json.load(f)
except Exception as e:
    print(f"[warn] Model not loaded yet — run train.py first. ({e})")

if os.path.exists("disease_info.json"):
    with open("disease_info.json") as f:
        disease_info = json.load(f)


def prepare_image(path):
    img = Image.open(path).convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    file.save(save_path)

    try:
        x = prepare_image(save_path)
        preds = model.predict(x)[0]
        top_idx = int(np.argmax(preds))
        confidence = float(preds[top_idx])
        label = class_names[top_idx]

        info = disease_info.get(label, {
            "description": "No details available for this class yet.",
            "treatment": "Consult a local agricultural extension office.",
        })

        return jsonify({
            "class": label,
            "confidence": round(confidence * 100, 2),
            "description": info["description"],
            "treatment": info["treatment"],
            "image_url": f"/{save_path}",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
