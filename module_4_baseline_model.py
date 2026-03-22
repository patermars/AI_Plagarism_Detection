import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder


def train_logistic(train_feats, y_train, max_iter=1000, C=1.0):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(y_train)
    lr_model = LogisticRegression(
        C=C, max_iter=max_iter, solver="saga", class_weight="balanced", n_jobs=-1,
    )
    lr_model.fit(train_feats, encoded_labels)
    return lr_model, encoder


def evaluate_model(model, feats, y_true, encoder, model_name="Model"):
    if hasattr(feats, "toarray"):
        X = feats.toarray()
    else:
        X = feats
    encoded_true = encoder.transform(y_true)
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]
    print(f"\n{'='*40}")
    print(f"{model_name} results")
    print(classification_report(encoded_true, predictions, target_names=encoder.classes_))
    auc = roc_auc_score(encoded_true, probabilities)
    print(f"ROC-AUC: {auc:.4f}")
    return auc


def load_model(model_path, encoder_path):
    return joblib.load(model_path), joblib.load(encoder_path)


if __name__ == "__main__":
    from module_1_data_prep import load_and_split, scrub_text
    from module_2_features import fit_tfidf, transform_features

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split("data.csv")

    # Apply heavy cleaning for TF-IDF
    X_train_tfidf = X_train.apply(scrub_text)
    X_val_tfidf = X_val.apply(scrub_text)

    vectorizer, train_feats = fit_tfidf(X_train_tfidf)
    val_feats = transform_features(vectorizer, X_val_tfidf)

    print("Training Logistic Regression...")
    lr_model, enc_lr = train_logistic(train_feats, y_train)
    evaluate_model(lr_model, val_feats, y_val, enc_lr, "Logistic Regression — Val")

    joblib.dump(lr_model, "lr_model.pkl")
    joblib.dump(enc_lr, "lr_encoder.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Saved: lr_model.pkl, lr_encoder.pkl, tfidf_vectorizer.pkl")