# preprocessing.py
# Cleans raw text from data.csv and produces stratified train / val / test splits
# (70 / 15 / 15) ready for downstream feature extraction and model training.

import pandas as pd
import re
from sklearn.model_selection import train_test_split


def clean_text(raw):
    # Strips HTML tags and URLs, then collapses whitespace.
    # Preserves punctuation and casing — both are meaningful features for DistilBERT.
    # Args: raw (str) — unprocessed text string
    # Returns: str — lightly cleaned text
    no_html = re.sub(r"<[^>]+>", " ", raw)
    no_urls = re.sub(r"http\S+|www\S+", " ", no_html)
    collapsed = re.sub(r"\s+", " ", no_urls).strip()
    return collapsed


def load_and_split(csv_path, seed=42):
    # Reads data.csv, cleans text, drops short entries, and returns stratified splits.
    # Split ratio: 70% train, 15% val, 15% test — stratified on author_type.
    # Args: csv_path (str) — path to data.csv; seed (int) — random seed for reproducibility
    # Returns: tuple of six pd.Series — (X_train, X_val, X_test, y_train, y_val, y_test)
    raw_df = pd.read_csv(csv_path)
    raw_df = raw_df.dropna(subset=["content_text", "author_type"])
    raw_df["clean_text"] = raw_df["content_text"].apply(clean_text)
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
    # Prints class distribution and a sample entry from the training set — quick sanity check.
    # Args: X_train (pd.Series) — training texts; y_train (pd.Series) — training labels
    summary = pd.Series(y_train).value_counts()
    print("Train label distribution:\n", summary)
    print("\nSample entry:\n", X_train.iloc[0])


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split("data.csv")
    peek(X_train, y_train)
    print(f"\nSizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
