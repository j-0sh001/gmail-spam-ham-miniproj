"""
train_model.py - Train and serialize the TF-IDF + Multinomial Naive Bayes spam/ham classifier.

NLP pipeline:
  1. Lexical tokenization        (TfidfVectorizer default word tokenizer)
  2. Case folding                (lowercase=True)
  3. Stopword elimination        (stop_words='english')
  4. Unigram vocabulary          (ngram_range=(1, 1))
  5. TF-IDF vectorization        (sublinear_tf=True)
  6. Supervised classification   (MultinomialNB)

Usage:
    python train_model.py                       # downloads the SMS dataset
    python train_model.py --data sms.tsv        # use a local copy (label<TAB>message)
"""
import argparse
import json
import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
ARTIFACT_DIR = "model"


def load_data(path_or_url: str) -> pd.DataFrame:
    df = pd.read_csv(path_or_url, sep="\t", header=None, names=["label", "message"])
    df = df.dropna().drop_duplicates()
    df["target"] = df["label"].map({"ham": 0, "spam": 1})
    return df.dropna(subset=["target"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DATA_URL, help="TSV path or URL (label<TAB>message)")
    parser.add_argument("--alpha", type=float, default=0.1, help="NB smoothing")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df)} messages | spam ratio: {df['target'].mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["target"],
        test_size=args.test_size, random_state=42, stratify=df["target"],
    )

    # Fit the vectorizer on training data only (no test leakage).
    vectorizer = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        sublinear_tf=True,
        ngram_range=(1, 1),
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = MultinomialNB(alpha=args.alpha)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),  # [[TN, FP], [FN, TP]]
        "vocab_size": len(vectorizer.vocabulary_),
        "train_size": int(X_train.shape[0]),
        "test_size": int(X_test.shape[0]),
    }

    print("\n--- Test set evaluation ---")
    for k in ("accuracy", "precision", "recall", "f1"):
        print(f"{k:>10}: {metrics[k]:.4f}")
    tn, fp, fn, tp = metrics["confusion_matrix"][0] + metrics["confusion_matrix"][1]
    print(f"\nConfusion matrix (rows = actual, cols = predicted)\n"
          f"             ham   spam\n  ham     {tn:6d} {fp:6d}\n  spam    {fn:6d} {tp:6d}")

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(ARTIFACT_DIR, "tfidf_vectorizer.joblib"))
    joblib.dump(model, os.path.join(ARTIFACT_DIR, "mnb_model.joblib"))
    with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved artifacts to ./{ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
