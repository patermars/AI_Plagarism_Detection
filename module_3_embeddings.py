import numpy as np
from sentence_transformers import SentenceTransformer
import joblib


SBERT_MODEL = "all-MiniLM-L6-v2"


def load_embedder(model_name=SBERT_MODEL):
    embedder = SentenceTransformer(model_name)
    return embedder


def encode_texts(embedder, texts, batch_size=64, show_bar=True):
    text_list = list(texts)
    vectors = embedder.encode(
        text_list,
        batch_size=batch_size,
        show_progress_bar=show_bar,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return vectors.astype(np.float32)


def save_embeddings(vectors, path):
    np.save(path, vectors)


def load_embeddings(path):
    return np.load(path)


if __name__ == "__main__":
    from module_1_data_prep import load_and_split

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = load_and_split("data.csv")

    embedder = load_embedder()

    print("Encoding training set...")
    train_vecs = encode_texts(embedder, X_train)
    print("Encoding validation set...")
    val_vecs = encode_texts(embedder, X_val)
    print("Encoding test set...")
    test_vecs = encode_texts(embedder, X_test)

    save_embeddings(train_vecs, "train_embeddings.npy")
    save_embeddings(val_vecs, "val_embeddings.npy")
    save_embeddings(test_vecs, "test_embeddings.npy")

    print(f"Embedding shape: {train_vecs.shape}")
