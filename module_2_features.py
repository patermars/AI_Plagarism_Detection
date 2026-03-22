import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


def fit_tfidf(train_texts, max_feats=30000, ngram_range=(1, 2)):
    vectorizer = TfidfVectorizer(
        max_features=max_feats,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )
    tfidf_matrix = vectorizer.fit_transform(train_texts)
    return vectorizer, tfidf_matrix


def transform_features(vectorizer, texts):
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path="tfidf_vectorizer.pkl"):
    joblib.dump(vectorizer, path)


def load_vectorizer(path="tfidf_vectorizer.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    from module_1_data_prep import load_and_split, scrub_text

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split("data.csv")

    # Apply heavy cleaning for TF-IDF (scrub_text strips punct/case)
    X_train_tfidf = X_train.apply(scrub_text)
    X_val_tfidf = X_val.apply(scrub_text)

    vectorizer, train_feats = fit_tfidf(X_train_tfidf)
    val_feats = transform_features(vectorizer, X_val_tfidf)

    print("Train feature matrix shape:", train_feats.shape)
    print("Val feature matrix shape  :", val_feats.shape)
    save_vectorizer(vectorizer)
