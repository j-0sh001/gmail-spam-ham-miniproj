"""
app.py - Flask inference API for the Gmail spam/ham extension.

Endpoints:
  GET  /health         -> {"status": "ok", ...model metrics}
  POST /predict        -> {"text": "..."}               -> single prediction
  POST /predict_batch  -> {"texts": ["...", "..."]}     -> list of predictions

Run:  python app.py
"""
import json
import os

import joblib
from flask import Flask, jsonify, request

ARTIFACT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
SPAM_THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", 0.5))
MAX_TEXTS = 200
MAX_CHARS = 5000

vectorizer = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
model = joblib.load(os.path.join(ARTIFACT_DIR, "mnb_model.joblib"))
SPAM_IDX = list(model.classes_).index(1)

try:
    with open(os.path.join(ARTIFACT_DIR, "metrics.json")) as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = {}

app = Flask(__name__)


def classify(texts):
    texts = [(t or "")[:MAX_CHARS] for t in texts]
    probs = model.predict_proba(vectorizer.transform(texts))[:, SPAM_IDX]
    results = []
    for p in probs:
        is_spam = bool(p >= SPAM_THRESHOLD)
        results.append({
            "label": "SPAM" if is_spam else "HAM",
            "spam_probability": round(float(p), 4),
            "ham_probability": round(float(1 - p), 4),
            "confidence": round(float(p if is_spam else 1 - p), 4),
        })
    return results


@app.get("/health")
def health():
    return jsonify({"status": "ok", "threshold": SPAM_THRESHOLD, "metrics": METRICS})


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text")
    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "'text' must be a non-empty string"}), 400
    return jsonify(classify([text])[0])


@app.post("/predict_batch")
def predict_batch():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts")
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        return jsonify({"error": "'texts' must be a list of strings"}), 400
    if len(texts) > MAX_TEXTS:
        return jsonify({"error": f"max {MAX_TEXTS} texts per request"}), 400
    return jsonify({"results": classify(texts)})


if __name__ == "__main__":
    # Bind to loopback only: the API is for the local extension, not the network.
    app.run(host="127.0.0.1", port=5000)
