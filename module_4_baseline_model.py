import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


def train_logistic(train_feats, y_train, max_iter=1000, C=1.0):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(y_train)
    lr_model = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="saga",
        class_weight="balanced",
        n_jobs=-1,
    )
    lr_model.fit(train_feats, encoded_labels)
    return lr_model, encoder


def train_xgboost(train_feats, y_train):
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(y_train)
    if hasattr(train_feats, "toarray"):
        X = train_feats.toarray()
    else:
        X = train_feats
    xgb_model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
        random_state=42,
    )
    xgb_model.fit(X, encoded_labels)
    return xgb_model, encoder


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


def save_model(model, encoder, model_path, encoder_path):
    joblib.dump(model, model_path)
    joblib.dump(encoder, encoder_path)


def load_model(model_path, encoder_path):
    model = joblib.load(model_path)
    encoder = joblib.load(encoder_path)
    return model, encoder


if __name__ == "__main__":
    from module_1_data_prep import load_and_split
    from module_2_features import fit_tfidf, transform_features

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = load_and_split("data.csv")

    vectorizer, train_feats = fit_tfidf(X_train)
    train_feats = transform_features(vectorizer, X_train, meta_train)
    val_feats = transform_features(vectorizer, X_val, meta_val)
    test_feats = transform_features(vectorizer, X_test, meta_test)

    lr_model, enc_lr = train_logistic(train_feats, y_train)
    evaluate_model(lr_model, val_feats, y_val, enc_lr, "Logistic Regression — Val")

    xgb_model, enc_xgb = train_xgboost(train_feats, y_train)
    evaluate_model(xgb_model, val_feats, y_val, enc_xgb, "XGBoost — Val")

    save_model(lr_model, enc_lr, "lr_model.pkl", "lr_encoder.pkl")
    save_model(xgb_model, enc_xgb, "xgb_model.pkl", "xgb_encoder.pkl")
