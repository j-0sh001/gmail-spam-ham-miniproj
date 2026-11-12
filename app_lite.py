"""
app_lite.py - Standard-library-only inference server (no pip install needed).

Reproduces the scikit-learn pipeline exactly from model/model.json:
  tokenize (regex) -> lowercase -> drop out-of-vocabulary / stopwords
  -> sublinear TF * IDF -> L2 normalize -> Multinomial NB posterior.

Same API as app.py:
  GET  /health
  POST /predict        {"text": "..."}
  POST /predict_batch  {"texts": ["...", ...]}

Run:  python app_lite.py
"""
import json
import math
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
SPAM_THRESHOLD = float(os.environ.get("SPAM_THRESHOLD", 0.5))
MAX_TEXTS = 200
MAX_CHARS = 5000

with open(os.path.join(HERE, "model", "model.json")) as f:
    M = json.load(f)
try:
    with open(os.path.join(HERE, "model", "metrics.json")) as f:
        METRICS = json.load(f)
except FileNotFoundError:
    METRICS = {}

TOKEN_RE = re.compile(M["token_pattern"])
VOCAB = M["vocabulary"]
IDF = M["idf"]
PRIOR = M["class_log_prior"]
FLP = M["feature_log_prob"]
SPAM = M["spam_index"]


def spam_probability(text: str) -> float:
    text = text[:MAX_CHARS]
    if M["lowercase"]:
        text = text.lower()
    counts = {}
    for tok in TOKEN_RE.findall(text):
        idx = VOCAB.get(tok)
        if idx is not None:
            counts[idx] = counts.get(idx, 0) + 1
    weights = {}
    for idx, tf in counts.items():
        tf = 1 + math.log(tf) if M["sublinear_tf"] else tf
        weights[idx] = tf * IDF[idx]
    norm = math.sqrt(sum(w * w for w in weights.values()))
    if norm > 0:
        weights = {i: w / norm for i, w in weights.items()}
    scores = [PRIOR[c] + sum(w * FLP[c][i] for i, w in weights.items()) for c in range(len(PRIOR))]
    m = max(scores)
    exps = [math.exp(s - m) for s in scores]
    return exps[SPAM] / sum(exps)


def classify(texts):
    out = []
    for t in texts:
        p = spam_probability(t or "")
        is_spam = p >= SPAM_THRESHOLD
        out.append({
            "label": "SPAM" if is_spam else "HAM",
            "spam_probability": round(p, 4),
            "ham_probability": round(1 - p, 4),
            "confidence": round(p if is_spam else 1 - p, 4),
        })
    return out


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            self._send(200, {"status": "ok", "threshold": SPAM_THRESHOLD, "metrics": METRICS})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        data = self._json_body()
        if self.path == "/predict":
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                return self._send(400, {"error": "'text' must be a non-empty string"})
            self._send(200, classify([text])[0])
        elif self.path == "/predict_batch":
            texts = data.get("texts")
            if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                return self._send(400, {"error": "'texts' must be a list of strings"})
            if len(texts) > MAX_TEXTS:
                return self._send(400, {"error": f"max {MAX_TEXTS} texts per request"})
            self._send(200, {"results": classify(texts)})
        else:
            self._send(404, {"error": "not found"})

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("Serving on http://127.0.0.1:5000  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", 5000), Handler).serve_forever()
