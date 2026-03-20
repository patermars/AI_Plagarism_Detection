import re
import numpy as np
import torch
import joblib
from module_1_data_prep import scrub_text
from module_2_features import load_vectorizer, transform_features
from module_5_bert_finetune import load_bert, SAVE_DIR
from module_6_ensemble import get_bert_probs, load_ensemble_weights


def split_into_sentences(paragraph):
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in parts if len(s.strip()) > 10]


def get_lr_probs(lr_model, vectorizer, texts):
    feats = transform_features(vectorizer, list(texts))
    return lr_model.predict_proba(feats)[:, 1]


def score_paragraph(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w):
    cleaned = scrub_text(paragraph)
    lr_prob = get_lr_probs(lr_model, vectorizer, [cleaned])[0]
    bert_prob = get_bert_probs(bert_model, tokenizer, [cleaned], device)[0]
    return float(lr_w * lr_prob + bert_w * bert_prob)


def score_sentences(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w):
    sentences = split_into_sentences(paragraph)
    if not sentences:
        return []
    lr_probs = get_lr_probs(lr_model, vectorizer, sentences)
    bert_probs = get_bert_probs(bert_model, tokenizer, [scrub_text(s) for s in sentences], device)
    blended = lr_w * lr_probs + bert_w * bert_probs
    return [{"sentence": s, "ai_probability": round(float(p), 4)} for s, p in zip(sentences, blended)]


def format_report(paragraph_score, sentence_scores, cutoff=0.5):
    verdict = "AI-generated" if paragraph_score >= cutoff else "Human-written"
    print(f"\n{'='*55}")
    print(f"Verdict      : {verdict}")
    print(f"AI confidence: {round(paragraph_score * 100, 1)}%")
    print(f"{'='*55}")
    print("\nSentence breakdown:")
    for item in sentence_scores:
        flag = "  [AI]" if item["ai_probability"] >= cutoff else "      "
        print(f"{flag} {round(item['ai_probability']*100,1):5.1f}%  {item['sentence'][:90]}")
    print()
    return {"verdict": verdict, "paragraph_ai_score": paragraph_score, "sentences": sentence_scores}


def load_pipeline(vectorizer_path="tfidf_vectorizer.pkl",
                  lr_model_path="lr_model.pkl",
                  bert_dir=SAVE_DIR):
    vectorizer = load_vectorizer(vectorizer_path)
    lr_model = joblib.load(lr_model_path)
    bert_model, tokenizer, _ = load_bert(bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    lr_w, bert_w = 0.3, 0.7
    return vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w


def run(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w):
    paragraph_score = score_paragraph(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w)
    sentence_scores = score_sentences(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w)
    return format_report(paragraph_score, sentence_scores)


if __name__ == "__main__":
    sample = (
        "The mitochondria are the powerhouse of the cell. They generate ATP through oxidative "
        "phosphorylation and play a central role in apoptosis. Recent studies have demonstrated "
        "that mitochondrial dysfunction is implicated in a wide range of human diseases including "
        "Parkinson's and Alzheimer's. The cell membrane regulates what enters and exits the cell."
    )
    pipeline = load_pipeline()
    run(sample, *pipeline)