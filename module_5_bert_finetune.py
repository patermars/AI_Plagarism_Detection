import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, classification_report
import os


BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SAVE_DIR = "distilbert_detector"


class ParagraphDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_SEQ_LEN):
        self.texts = list(texts)
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_data_loaders(X_train, y_train, X_val, y_val, tokenizer, encoder):
    train_labels = encoder.transform(y_train)
    val_labels = encoder.transform(y_val)
    train_set = ParagraphDataset(X_train, train_labels, tokenizer)
    val_set = ParagraphDataset(X_val, val_labels, tokenizer)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return train_loader, val_loader


def train_bert(X_train, X_val, y_train, y_val):
    device = get_device()
    print(f"Using device: {device}")

    encoder = LabelEncoder()
    encoder.fit(y_train)
    num_labels = len(encoder.classes_)

    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_labels
    ).to(device)

    train_loader, val_loader = build_data_loaders(X_train, y_train, X_val, y_val, tokenizer, encoder)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = output.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        val_auc = evaluate_bert(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} — loss: {avg_loss:.4f} — val AUC: {val_auc:.4f}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    import joblib
    joblib.dump(encoder, os.path.join(SAVE_DIR, "label_encoder.pkl"))
    print(f"Model saved to {SAVE_DIR}/")

    return model, tokenizer, encoder


def evaluate_bert(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].numpy()
            output = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels)
    return roc_auc_score(all_labels, all_probs)


def load_bert(save_dir=SAVE_DIR):
    import joblib
    tokenizer = DistilBertTokenizerFast.from_pretrained(save_dir)
    model = DistilBertForSequenceClassification.from_pretrained(save_dir)
    encoder = joblib.load(os.path.join(save_dir, "label_encoder.pkl"))
    return model, tokenizer, encoder


if __name__ == "__main__":
    from module_1_data_prep import load_and_split

    X_train, X_val, X_test, y_train, y_val, y_test, meta_train, meta_val, meta_test = load_and_split("data.csv")
    model, tokenizer, encoder = train_bert(X_train, X_val, y_train, y_val)
