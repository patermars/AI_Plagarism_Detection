"""
evaluate.py — Performance metrics for the DistilBERT AI plagiarism detector.

Evaluates on the held-out TEST set and outputs:
  - Classification report (precision / recall / F1 / support)
  - ROC-AUC, accuracy, MCC
  - Confusion matrix (printed + saved as PNG)
"""

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    matthews_corrcoef,
)

from preprocessing import load_and_split
from bert_finetune import load_bert, SAVE_DIR
from inference import get_bert_probs


def _print_header(title):
    # Prints a visually distinct section header to stdout — keeps console output scannable.
    # Args: title (str)
    w = 60
    print(f"\n{'━' * w}")
    print(f"  {title}")
    print(f"{'━' * w}")


def main():
    # Loads the held-out test split and the saved DistilBERT model, runs inference,
    # then prints classification report / scalar metrics and saves roc_curve.png
    # and confusion_matrix.png to the working directory.
    _print_header("Loading data & model")

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split("data.csv")
    print(f"  Test samples: {len(X_test)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model, tokenizer, encoder = load_bert(SAVE_DIR)
    bert_model = bert_model.to(device)
    class_names = list(encoder.classes_)
    print(f"  Classes     : {class_names}")
    print(f"  Device      : {device}")

    y_true = encoder.transform(y_test)

    _print_header("Running inference on test set")
    ai_probs = get_bert_probs(bert_model, tokenizer, X_test, device)
    preds = (ai_probs < 0.5).astype(int)
    human_probs = 1.0 - ai_probs

    _print_header("DistilBERT Results")
    print(classification_report(y_true, preds, target_names=class_names, digits=4))

    cm = confusion_matrix(y_true, preds)
    col_w = max(len(c) for c in class_names) + 2
    header = " " * (col_w + 4) + "".join(f"{c:>{col_w}}" for c in class_names)
    print("  Confusion Matrix:")
    print(header)
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:>{col_w}}" for v in row)
        print(f"  {class_names[i]:<{col_w}}{row_str}")
    print()

    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, human_probs)
    f1_m = f1_score(y_true, preds, average="macro")
    f1_w = f1_score(y_true, preds, average="weighted")
    prec = precision_score(y_true, preds, average="macro")
    rec = recall_score(y_true, preds, average="macro")
    mcc = matthews_corrcoef(y_true, preds)

    print(f"  Accuracy           : {acc:.4f}")
    print(f"  ROC-AUC            : {auc:.4f}")
    print(f"  F1 (macro)         : {f1_m:.4f}")
    print(f"  F1 (weighted)      : {f1_w:.4f}")
    print(f"  Precision (macro)  : {prec:.4f}")
    print(f"  Recall (macro)     : {rec:.4f}")
    print(f"  MCC                : {mcc:.4f}")

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix — Test Set", fontsize=12, fontweight="bold")
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=13,
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✔ Confusion matrix saved → confusion_matrix.png")

    _print_header("Summary")
    print(f"  ROC-AUC : {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  MCC     : {mcc:.4f}")
    print(f"  Samples : {len(y_test)}\n")


if __name__ == "__main__":
    main()
