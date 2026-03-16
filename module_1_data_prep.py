import pandas as pd
import re
from sklearn.model_selection import train_test_split


def scrub_text(raw):
    lowered = raw.lower()
    no_html = re.sub(r"<[^>]+>", " ", lowered)
    no_urls = re.sub(r"http\S+|www\S+", " ", no_html)
    no_punct = re.sub(r"[^a-z0-9\s]", " ", no_urls)
    collapsed = re.sub(r"\s+", " ", no_punct).strip()
    return collapsed


def load_and_split(csv_path, seed=42):
    raw_df = pd.read_csv(csv_path)

    text_col = "content_text"
    label_col = "author_type"

    raw_df = raw_df.dropna(subset=[text_col, label_col])
    raw_df["clean_text"] = raw_df[text_col].apply(scrub_text)
    raw_df = raw_df[raw_df["clean_text"].str.len() > 20].reset_index(drop=True)

    X = raw_df["clean_text"]
    y = raw_df[label_col]
    meta = raw_df[[
        "perplexity_score",
        "burstiness_index",
        "syntactic_variability",
        "semantic_coherence_score",
        "lexical_diversity_ratio",
        "readability_grade_level",
        "generation_confidence_score",
    ]].reset_index(drop=True)

    X_train, X_holdout, y_train, y_holdout, meta_train, meta_holdout = train_test_split(
        X, y, meta, test_size=0.30, stratify=y, random_state=seed
    )
    X_val, X_test, y_val, y_test, meta_val, meta_test = train_test_split(
        X_holdout, y_holdout, meta_holdout, test_size=0.50, stratify=y_holdout, random_state=seed
    )

    return (
        X_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_val.reset_index(drop=True),
        y_test.reset_index(drop=True),
        meta_train.reset_index(drop=True),
        meta_val.reset_index(drop=True),
        meta_test.reset_index(drop=True),
    )


def peek(X_train, y_train):
    summary = pd.Series(y_train).value_counts()
    print("Train label distribution:\n", summary)
    print("\nSample entry:\n", X_train.iloc[0])


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = load_and_split("data.csv")
    peek(X_train, y_train)
    print(f"\nSizes — train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Meta features shape: {meta_train.shape}")
