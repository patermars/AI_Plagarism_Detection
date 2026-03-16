import numpy as np
import scipy.sparse as sp
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


def transform_features(vectorizer, texts, meta_df=None):
    tfidf_part = vectorizer.transform(texts)
    if meta_df is not None:
        dense_meta = sp.csr_matrix(meta_df.values.astype(np.float32))
        combined = sp.hstack([tfidf_part, dense_meta])
        return combined
    return tfidf_part


def save_vectorizer(vectorizer, path="tfidf_vectorizer.pkl"):
    joblib.dump(vectorizer, path)


def load_vectorizer(path="tfidf_vectorizer.pkl"):
    return joblib.load(path)


if __name__ == "__main__":
    from module_1_data_prep import load_and_split

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = load_and_split("data.csv")

    vectorizer, train_feats = fit_tfidf(X_train)
    val_feats = transform_features(vectorizer, X_val, meta_val)
    train_feats_full = transform_features(vectorizer, X_train, meta_train)

    print("Train feature matrix shape:", train_feats_full.shape)
    print("Val feature matrix shape  :", val_feats.shape)
    print(f"Meta columns used         : {meta_train.shape[1]}")
    save_vectorizer(vectorizer)
