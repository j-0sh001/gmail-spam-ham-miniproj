"""
export_model.py - Export the fitted scikit-learn TF-IDF vectorizer + MultinomialNB
into a plain JSON file so app_lite.py can serve predictions with the Python
standard library only (no numpy / pandas / scikit-learn on the serving machine).

Run after train_model.py:   python export_model.py
"""
import json
import os

import joblib

ARTIFACT_DIR = "model"

vec = joblib.load(os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
nb = joblib.load(os.path.join(ARTIFACT_DIR, "mnb_model.joblib"))
spam_idx = list(nb.classes_).index(1)

payload = {
    "token_pattern": vec.token_pattern,          # lexical tokenization
    "lowercase": vec.lowercase,                  # case folding
    "sublinear_tf": vec.sublinear_tf,
    "vocabulary": {t: int(i) for t, i in vec.vocabulary_.items()},  # stopwords already excluded
    "idf": [float(x) for x in vec.idf_],
    "class_log_prior": [float(x) for x in nb.class_log_prior_],
    "feature_log_prob": [[float(x) for x in row] for row in nb.feature_log_prob_],
    "spam_index": spam_idx,
}

with open(os.path.join(ARTIFACT_DIR, "model.json"), "w") as f:
    json.dump(payload, f)
print("Wrote model/model.json")
