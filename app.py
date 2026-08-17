import json
import os
import uuid

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = os.path.join(
    "models",
    "agrisense_model.tflite"
)

UPLOAD_DIR = os.path.join(
    "static",
    "uploads"
)

IMG_SIZE = (224, 224)


# --------------------------------------------------
# Flask application
# --------------------------------------------------

app = Flask(__name__)

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Variables
# --------------------------------------------------

interpreter = None
input_details = None
output_details = None

class_names = []
disease_info = {}


# --------------------------------------------------
# Load TFLite model
# --------------------------------------------------

try:

    interpreter = Interpreter(
        model_path=MODEL_PATH
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("[INFO] TFLite model loaded successfully.")

    print(
        "[INFO] Input shape:",
        input_details[0]["shape"]
    )

    print(
        "[INFO] Output shape:",
        output_details[0]["shape"]
    )

except Exception as e:

    print(
        f"[ERROR] TFLite model could not be loaded: {e}"
    )


# --------------------------------------------------
# Load class names
# --------------------------------------------------

try:

    with open(
        os.path.join(
            "models",
            "class_names.json"
        ),
        "r"
    ) as f:

        class_names = json.load(f)

    print(
        "[INFO] Number of classes:",
        len(class_names)
    )

except Exception as e:

    print(
        f"[ERROR] Could not load class_names.json: {e}"
    )


# --------------------------------------------------
# Load disease information
# --------------------------------------------------

if os.path.exists("disease_info.json"):

    try:

        with open(
            "disease_info.json",
            "r"
        ) as f:

            disease_info = json.load(f)

        print(
            "[INFO] Disease information loaded successfully."
        )

    except Exception as e:

        print(
            f"[ERROR] Could not load disease_info.json: {e}"
        )


# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------

def prepare_image(path):

    img = Image.open(
        path
    ).convert("RGB")

    img = img.resize(
        IMG_SIZE
    )

    arr = np.array(
        img,
        dtype=np.float32
    )

    # MobileNetV2 preprocessing:
    # Convert [0, 255] → [-1, 1]

    arr = (arr / 127.5) - 1.0

    arr = np.expand_dims(
        arr,
        axis=0
    )

    return arr


# --------------------------------------------------
# Home page
# --------------------------------------------------

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# --------------------------------------------------
# Prediction API
# --------------------------------------------------

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    print(
        "[INFO] /predict request received."
    )

    # Check model
    if interpreter is None:

        return jsonify({
            "error": "TFLite model is not loaded."
        }), 503


    # Check uploaded image
    if "image" not in request.files:

        return jsonify({
            "error": "No image uploaded."
        }), 400


    file = request.files["image"]


    # Check filename
    if file.filename == "":

        return jsonify({
            "error": "No image selected."
        }), 400


    # Create unique filename
    filename = (
        f"{uuid.uuid4().hex}_"
        f"{file.filename}"
    )

    save_path = os.path.join(
        UPLOAD_DIR,
        filename
    )


    try:

        # --------------------------------------------------
        # Save image
        # --------------------------------------------------

        file.save(
            save_path
        )

        print(
            "[INFO] Image saved:",
            save_path
        )


        # --------------------------------------------------
        # Prepare image
        # --------------------------------------------------

        x = prepare_image(
            save_path
        )

        print(
            "[INFO] Image prepared."
        )


        # --------------------------------------------------
        # TFLite prediction
        # --------------------------------------------------

        print(
            "[INFO] Starting TFLite prediction..."
        )

        interpreter.set_tensor(
            input_details[0]["index"],
            x
        )

        interpreter.invoke()

        preds = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        print(
            "[INFO] TFLite prediction completed."
        )


        # --------------------------------------------------
        # Find highest probability
        # --------------------------------------------------

        top_idx = int(
            np.argmax(preds)
        )

        confidence = float(
            preds[top_idx]
        )


        # Check class index
        if top_idx >= len(
            class_names
        ):

            raise ValueError(
                "Prediction class index is "
                "outside the class list."
            )


        # Get class name
        label = class_names[
            top_idx
        ]


        print(
            f"[INFO] Prediction: "
            f"{label} "
            f"({confidence * 100:.2f}%)"
        )


        # --------------------------------------------------
        # Disease information
        # --------------------------------------------------

        info = disease_info.get(
            label,
            {
                "description":
                    "No details available "
                    "for this class yet.",

                "treatment":
                    "Consult a local "
                    "agricultural extension office."
            }
        )


        # --------------------------------------------------
        # JSON response
        # --------------------------------------------------

        response = {

            "class": label,

            "confidence": round(
                confidence * 100,
                2
            ),

            "description": info.get(
                "description",
                "No description available."
            ),

            "treatment": info.get(
                "treatment",
                "Consult a local agricultural "
                "extension office."
            ),

            "image_url": (
                f"/{save_path}"
            )
        }


        print(
            "[INFO] Sending prediction response."
        )


        return jsonify(
            response
        ), 200


    except Exception as e:

        print(
            f"[ERROR] Prediction failed: {e}"
        )

        return jsonify({
            "error": str(e)
        }), 500


# --------------------------------------------------
# Run application
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )