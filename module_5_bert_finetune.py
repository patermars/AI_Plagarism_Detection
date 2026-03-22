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
from sklearn.metrics import roc_auc_score
import os
import time


BERT_MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LEN = 256
BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
SAVE_DIR = "distilbert_detector"
LOG_EVERY = 50


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
    print(f"\n{'='*60}")
    print(f"Device        : {device}")
    print(f"Train samples : {len(X_train)}")
    print(f"Val samples   : {len(X_val)}")
    print(f"Epochs        : {EPOCHS}")
    print(f"Batch size    : {BATCH_SIZE}")
    print(f"Max seq len   : {MAX_SEQ_LEN}")
    print(f"{'='*60}\n")

    encoder = LabelEncoder()
    encoder.fit(y_train)
    num_labels = len(encoder.classes_)
    print(f"Labels: {list(encoder.classes_)}")

    print("Loading tokenizer and model...")
    tokenizer = DistilBertTokenizerFast.from_pretrained(BERT_MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=num_labels
    ).to(device)
    print("Model loaded.\n")

    train_loader, val_loader = build_data_loaders(X_train, y_train, X_val, y_val, tokenizer, encoder)
    total_batches = len(train_loader)
    print(f"Batches per epoch: {total_batches}")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    total_steps = total_batches * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"Total steps: {total_steps}  |  Warmup steps: {warmup_steps}\n")

    overall_start = time.time()

    for epoch in range(EPOCHS):
        print(f"{'─'*60}")
        print(f"Epoch {epoch+1}/{EPOCHS} starting...")
        print(f"{'─'*60}")
        model.train()
        running_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            running_loss += output.loss.item()

            if batch_idx % LOG_EVERY == 0 or batch_idx == total_batches:
                avg_loss = running_loss / batch_idx
                elapsed = time.time() - epoch_start
                batches_left = total_batches - batch_idx
                eta_secs = (elapsed / batch_idx) * batches_left
                print(
                    f"  Batch {batch_idx:>4}/{total_batches}"
                    f"  |  avg loss: {avg_loss:.4f}"
                    f"  |  elapsed: {elapsed/60:.1f}m"
                    f"  |  ETA: {eta_secs/60:.1f}m"
                )

        epoch_loss = running_loss / total_batches
        epoch_time = time.time() - epoch_start
        print(f"\nRunning validation for epoch {epoch+1}...")
        val_auc = evaluate_bert(model, val_loader, device)
        print(f"Epoch {epoch+1}/{EPOCHS} complete")
        print(f"  Loss : {epoch_loss:.4f}")
        print(f"  AUC  : {val_auc:.4f}")
        print(f"  Time : {epoch_time/60:.1f}m\n")

    total_time = time.time() - overall_start
    print(f"{'='*60}")
    print(f"Training complete in {total_time/60:.1f}m")
    print(f"{'='*60}\n")

    print(f"Saving model to {SAVE_DIR}/...")
    os.makedirs(SAVE_DIR, exist_ok=True)
    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    import joblib
    joblib.dump(encoder, os.path.join(SAVE_DIR, "label_encoder.pkl"))
    print("Saved.\n")

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

    X_train, X_val, X_test, y_train, y_val, y_test = load_and_split("data.csv")
    train_bert(X_train, X_val, y_train, y_val)
