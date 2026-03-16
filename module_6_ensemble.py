import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report
import joblib


def get_xgb_probs(xgb_model, feats):
    if hasattr(feats, "toarray"):
        X = feats.toarray()
    else:
        X = feats
    return xgb_model.predict_proba(X)[:, 1]


def get_bert_probs(bert_model, tokenizer, texts, device, batch_size=32, max_len=256):
    bert_model.eval()
    all_probs = []
    text_list = list(texts)
    for start in range(0, len(text_list), batch_size):
        batch_texts = text_list[start: start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs, dtype=np.float32)


def soft_vote(xgb_probs, bert_probs, xgb_weight=0.4, bert_weight=0.6):
    return xgb_weight * xgb_probs + bert_weight * bert_probs


def threshold_predictions(blended_probs, cutoff=0.5):
    return (blended_probs >= cutoff).astype(int)


def evaluate_ensemble(blended_probs, y_true, encoder):
    true_encoded = encoder.transform(y_true)
    preds = threshold_predictions(blended_probs)
    auc = roc_auc_score(true_encoded, blended_probs)
    print("\nEnsemble evaluation")
    print(classification_report(true_encoded, preds, target_names=encoder.classes_))
    print(f"ROC-AUC: {auc:.4f}")
    return auc


def save_ensemble_weights(xgb_weight, bert_weight, path="ensemble_weights.pkl"):
    joblib.dump({"xgb": xgb_weight, "bert": bert_weight}, path)


def load_ensemble_weights(path="ensemble_weights.pkl"):
    weights = joblib.load(path)
    return weights["xgb"], weights["bert"]


def tune_ensemble_weights(xgb_probs_val, bert_probs_val, y_val, encoder, steps=20):
    best_auc = 0.0
    best_xgb_w = 0.4
    encoded_val = encoder.transform(y_val)
    for step in range(steps + 1):
        xgb_w = step / steps
        bert_w = 1.0 - xgb_w
        blended = soft_vote(xgb_probs_val, bert_probs_val, xgb_w, bert_w)
        auc = roc_auc_score(encoded_val, blended)
        if auc > best_auc:
            best_auc = auc
            best_xgb_w = xgb_w
    best_bert_w = 1.0 - best_xgb_w
    print(f"Best weights — XGB: {best_xgb_w:.2f}, BERT: {best_bert_w:.2f}, AUC: {best_auc:.4f}")
    return best_xgb_w, best_bert_w


if __name__ == "__main__":
    from module_1_data_prep import load_and_split
    from module_2_features import fit_tfidf, transform_features
    from module_4_baseline_model import load_model
    from module_5_bert_finetune import load_bert, SAVE_DIR

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = load_and_split("data.csv")

    vectorizer, _ = fit_tfidf(X_train)
    val_feats = transform_features(vectorizer, X_val, meta_val)
    test_feats = transform_features(vectorizer, X_test, meta_test)

    xgb_model, xgb_encoder = load_model("xgb_model.pkl", "xgb_encoder.pkl")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model, tokenizer, bert_encoder = load_bert(SAVE_DIR)
    bert_model = bert_model.to(device)

    xgb_val_probs = get_xgb_probs(xgb_model, val_feats)
    bert_val_probs = get_bert_probs(bert_model, tokenizer, X_val, device)

    best_xgb_w, best_bert_w = tune_ensemble_weights(xgb_val_probs, bert_val_probs, y_val, xgb_encoder)
    save_ensemble_weights(best_xgb_w, best_bert_w)

    xgb_test_probs = get_xgb_probs(xgb_model, test_feats)
    bert_test_probs = get_bert_probs(bert_model, tokenizer, X_test, device)
    final_probs = soft_vote(xgb_test_probs, bert_test_probs, best_xgb_w, best_bert_w)
    evaluate_ensemble(final_probs, y_test, xgb_encoder)
