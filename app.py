import os
import json
import numpy as np
from flask import Flask, jsonify, request, render_template

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(filename):
    candidates = [
        os.path.join(BASE_DIR, filename),
        os.path.join(BASE_DIR, "ANN_HAR_Project", filename),
        os.path.join(os.getcwd(), filename),
        os.path.join(os.getcwd(), "ANN_HAR_Project", filename)
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None

# Locate templates directory
template_dir = os.path.join(BASE_DIR, "templates")
if not os.path.isdir(template_dir):
    alt_template = os.path.join(BASE_DIR, "ANN_HAR_Project", "templates")
    if os.path.isdir(alt_template):
        template_dir = alt_template

app = Flask(__name__, template_folder=template_dir)
application = app
handler = app

CLASS_NAMES = ["Walking", "Walking Up", "Walking Down", "Sitting", "Standing", "Laying"]

# Lazy-loaded / cached model artifacts
_MODEL_DATA = None
_FEATURE_NAMES = None

def get_feature_names():
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        path = resolve_path("feature_names.json")
        if path and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                _FEATURE_NAMES = json.load(f)
        else:
            _FEATURE_NAMES = [f"Feature {i+1}" for i in range(561)]
    return _FEATURE_NAMES

def load_artifacts():
    global _MODEL_DATA
    if _MODEL_DATA is not None:
        return _MODEL_DATA

    weights_path = resolve_path("model_weights.npz")
    samples_path = resolve_path("test_samples.npz")

    if not weights_path or not samples_path:
        raise FileNotFoundError(
            f"Missing required model files. weights_path: {weights_path}, samples_path: {samples_path}"
        )

    w = np.load(weights_path)
    s = np.load(samples_path)

    _MODEL_DATA = {
        "w1": w["w1"],
        "b1": w["b1"],
        "w2": w["w2"],
        "b2": w["b2"],
        "w3": w["w3"],
        "b3": w["b3"],
        "mean": w["mean"],
        "scale": w["scale"],
        "X": s["X"],
        "y": s["y"]
    }
    return _MODEL_DATA

def predict_single(x_raw):
    data = load_artifacts()
    # Normalize features using saved StandardScaler mean & scale
    x_norm = (x_raw - data["mean"]) / data["scale"]

    # Layer 1: Dense 256 with ReLU
    h1 = np.maximum(0, np.dot(x_norm, data["w1"]) + data["b1"])

    # Layer 2: Dense 128 with ReLU
    h2 = np.maximum(0, np.dot(h1, data["w2"]) + data["b2"])

    # Layer 3: Dense 6 with Softmax
    logits = np.dot(h2, data["w3"]) + data["b3"]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / np.sum(exp_logits)

    pred_idx = int(np.argmax(probs))
    return CLASS_NAMES[pred_idx], probs


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "app": "Human-Activity-Guesser"})


@app.route("/api/stats")
def stats():
    data = load_artifacts()
    return jsonify({
        "total_samples": int(len(data["y"])),
        "classes": CLASS_NAMES,
        "n_features": int(data["X"].shape[1]),
        "test_accuracy": 0.9539
    })


@app.route("/api/sample")
def get_sample():
    data = load_artifacts()
    total = len(data["y"])
    try:
        index = int(request.args.get("index", 0))
    except (ValueError, TypeError):
        index = 0

    index = max(0, min(total - 1, index))

    x_raw = data["X"][index]
    actual_idx = int(data["y"][index])
    actual_name = CLASS_NAMES[actual_idx]

    pred_name, probs = predict_single(x_raw)
    feature_names = get_feature_names()

    features_preview = [
        {
            "index": i + 1,
            "name": feature_names[i] if i < len(feature_names) else f"Feature {i+1}",
            "value": float(x_raw[i])
        }
        for i in range(min(30, len(x_raw)))
    ]

    confidence = {
        CLASS_NAMES[i]: round(float(probs[i]), 5)
        for i in range(len(CLASS_NAMES))
    }

    return jsonify({
        "index": index,
        "total_samples": total,
        "actual": actual_name,
        "predicted": pred_name,
        "is_correct": bool(pred_name == actual_name),
        "confidence": confidence,
        "features": features_preview
    })


@app.route("/api/sample-by-activity")
def sample_by_activity():
    data = load_artifacts()
    act = request.args.get("activity", "Walking")
    if act in CLASS_NAMES:
        target_idx = CLASS_NAMES.index(act)
        matches = np.where(data["y"] == target_idx)[0]
        if len(matches) > 0:
            chosen = int(np.random.choice(matches))
            return jsonify({"index": chosen, "activity": act})
    return jsonify({"index": 0, "activity": CLASS_NAMES[0]})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
