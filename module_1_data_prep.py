import pandas as pd
import re
from sklearn.model_selection import train_test_split


def scrub_text(raw):
    """Heavy cleaning for TF-IDF: lowercase, strip HTML/URLs/punctuation."""
    lowered = raw.lower()
    no_html = re.sub(r"<[^>]+>", " ", lowered)
    no_urls = re.sub(r"http\S+|www\S+", " ", no_html)
    no_punct = re.sub(r"[^a-z0-9\s]", " ", no_urls)
    collapsed = re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def clean_for_bert(raw):
    """Light cleaning for BERT: keep punctuation and casing, only strip HTML/URLs."""
    no_html = re.sub(r"<[^>]+>", " ", raw)
    no_urls = re.sub(r"http\S+|www\S+", " ", no_html)
    collapsed = re.sub(r"\s+", " ", no_urls).strip()
    return collapsed


def load_and_split(csv_path, seed=42):
    raw_df = pd.read_csv(csv_path)
    raw_df = raw_df.dropna(subset=["content_text", "author_type"])
    raw_df["clean_text"] = raw_df["content_text"].apply(clean_for_bert)
    raw_df = raw_df[raw_df["clean_text"].str.len() > 20].reset_index(drop=True)

    X = raw_df["clean_text"]
    y = raw_df["author_type"]

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.50, stratify=y_holdout, random_state=seed
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


def peek(X_train, y_train):
    summary = pd.Series(y_train).value_counts()
    print("Train label distribution:\n", summary)
    print("\nSample entry:\n", X_train.iloc[0])


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split("data.csv")
    peek(X_train, y_train)
    print(f"\nSizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
