# traditional_models.py
# Trains and evaluates five traditional ML classifiers on TF-IDF features.
# Produces a side-by-side comparison table and saves results to traditional_results.csv.

import re
import string
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, matthews_corrcoef,
)
from preprocessing import load_and_split


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    "Linear SVM":          CalibratedClassifierCV(LinearSVC(max_iter=2000, C=1.0, random_state=42)),
    "Naive Bayes":         MultinomialNB(alpha=0.1),
    "Random Forest":       RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    "Gradient Boosting":   GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42),
}


def evaluate(y_true, y_pred, y_prob):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "F1 (macro)": round(f1_score(y_true, y_pred, average="macro"), 4),
        "Precision": round(precision_score(y_true, y_pred, average="macro"), 4),
        "Recall":    round(recall_score(y_true, y_pred, average="macro"), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4),
        "MCC":       round(matthews_corrcoef(y_true, y_pred), 4),
    }


def main():
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split("data.csv")

    # Combine train+val for traditional models (no need for a separate val set)
    X_fit = pd.concat([X_train, X_val]).reset_index(drop=True)
    y_fit = pd.concat([y_train, y_val]).reset_index(drop=True)

    print("Fitting TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=50_000, sublinear_tf=True, ngram_range=(1, 2))
    X_fit_vec = tfidf.fit_transform(X_fit)
    X_test_vec = tfidf.transform(X_test)

    # MultinomialNB requires non-negative features — TF-IDF with sublinear_tf is fine
    y_fit_enc  = (y_fit == "AI").astype(int).values
    y_test_enc = (y_test == "AI").astype(int).values

    rows = []
    for name, clf in MODELS.items():
        print(f"Training {name}...")
        clf.fit(X_fit_vec, y_fit_enc)
        y_pred = clf.predict(X_test_vec)
        y_prob = clf.predict_proba(X_test_vec)[:, 1]
        metrics = evaluate(y_test_enc, y_pred, y_prob)
        rows.append({"Model": name, **metrics})
        print(f"  Accuracy={metrics['Accuracy']}  AUC={metrics['ROC-AUC']}  MCC={metrics['MCC']}")

    results = pd.DataFrame(rows).set_index("Model")
    results.to_csv("traditional_results.csv")
    print("\n── Traditional Model Comparison ──────────────────────────")
    print(results.to_string())
    print(f"\nSaved → traditional_results.csv")

    # Bar chart
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(results))
    width = 0.15
    metrics_to_plot = ["Accuracy", "F1 (macro)", "ROC-AUC", "MCC"]
    for i, metric in enumerate(metrics_to_plot):
        ax.bar(x + i * width, results[metric], width, label=metric)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(results.index, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0.5, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Traditional Models — Test Set Performance")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("traditional_models_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved → traditional_models_comparison.png")

    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    print("Saved → tfidf_vectorizer.pkl")


if __name__ == "__main__":
    main()
