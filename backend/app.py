import os
import io
import json
import base64
import sqlite3
import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np

app = Flask(__name__)
CORS(app)

MODEL_PATH = "./model/traffic_sign_model.h5"
CLASS_NAMES_PATH = "./model/class_names.json"
ACCURACY_PATH = "./model/accuracy.txt"
DB_PATH = os.environ.get("DB_PATH", "./data/predictions.db")

model = None
class_names = {}
model_loaded = False


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            class_name TEXT,
            confidence REAL,
            image_base64 TEXT
        )
    """)
    conn.commit()
    conn.close()


def load_model():
    global model, class_names, model_loaded
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(MODEL_PATH)
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        model_loaded = True
        print("Model loaded successfully")
    except Exception as e:
        print(f"Model not loaded: {e}")
        model_loaded = False


def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((32, 32))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr, img


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_loaded": model_loaded})


@app.route("/model-info")
def model_info():
    accuracy = None
    if os.path.exists(ACCURACY_PATH):
        with open(ACCURACY_PATH, "r") as f:
            try:
                accuracy = float(f.read().strip())
            except Exception:
                accuracy = None
    return jsonify({
        "accuracy": accuracy,
        "model_loaded": model_loaded,
        "num_classes": len(class_names)
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return jsonify({"error": "Model not loaded. Please train the model first."}), 503
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    try:
        image_bytes = file.read()
        arr, img = preprocess_image(image_bytes)
        input_arr = np.expand_dims(arr, axis=0)
        predictions = model.predict(input_arr)
        pred_idx = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_idx])
        class_name = class_names.get(str(pred_idx), f"Class {pred_idx}")

        # Save thumbnail as base64
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        # Save to DB
        timestamp = datetime.datetime.now().isoformat()
        conn = sqlite3.connect(DB_PATH)
        conn.execute(
            "INSERT INTO predictions (timestamp, class_name, confidence, image_base64) VALUES (?,?,?,?)",
            (timestamp, class_name, confidence, img_b64)
        )
        conn.commit()
        conn.close()

        return jsonify({
            "class_id": pred_idx,
            "class_name": class_name,
            "confidence": round(confidence * 100, 2),
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 400


@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT id, timestamp, class_name, confidence, image_base64 FROM predictions ORDER BY id DESC LIMIT 20"
    ).fetchall()
    conn.close()
    result = [
        {
            "id": r[0],
            "timestamp": r[1],
            "class_name": r[2],
            "confidence": round(r[3] * 100, 2),
            "image_base64": r[4]
        }
        for r in rows
    ]
    return jsonify(result)


@app.route("/history", methods=["DELETE"])
def clear_history():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()
    return jsonify({"status": "cleared"})


def create_app():
    """Application factory — called by gunicorn and __main__."""
    init_db()
    load_model()
    return app

# Gunicorn entry point: gunicorn "app:create_app()"
# But we keep app as the wsgi callable for simplicity; run init at module level
# only once, guarded so import in train.py doesn't re-run it.
if __name__ != "__main__":
    # Running under gunicorn
    init_db()
    load_model()

if __name__ == "__main__":
    init_db()
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=False)
