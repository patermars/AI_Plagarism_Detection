import re
import numpy as np
import torch
import joblib
from module_1_data_prep import scrub_text
from module_2_features import load_vectorizer, transform_features
from module_5_bert_finetune import load_bert, SAVE_DIR
from module_6_ensemble import get_xgb_probs, get_bert_probs, soft_vote, load_ensemble_weights


def split_into_sentences(paragraph):
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    sentences = [s.strip() for s in parts if len(s.strip()) > 10]
    return sentences


def score_paragraph(paragraph, vectorizer, xgb_model, bert_model, tokenizer, device, xgb_w, bert_w):
    cleaned = scrub_text(paragraph)
    feats = transform_features(vectorizer, [cleaned])
    xgb_prob = get_xgb_probs(xgb_model, feats)[0]
    bert_prob = get_bert_probs(bert_model, tokenizer, [cleaned], device)[0]
    blended = float(xgb_w * xgb_prob + bert_w * bert_prob)
    return blended


def score_sentences(paragraph, vectorizer, xgb_model, bert_model, tokenizer, device, xgb_w, bert_w):
    sentences = split_into_sentences(paragraph)
    if not sentences:
        return []
    cleaned_sentences = [scrub_text(s) for s in sentences]
    feats = transform_features(vectorizer, cleaned_sentences)
    xgb_probs = get_xgb_probs(xgb_model, feats)
    bert_probs = get_bert_probs(bert_model, tokenizer, cleaned_sentences, device)
    blended = xgb_w * xgb_probs + bert_w * bert_probs
    results = []
    for original, score in zip(sentences, blended):
        results.append({"sentence": original, "ai_probability": round(float(score), 4)})
    return results


def format_report(paragraph_score, sentence_scores, cutoff=0.5):
    verdict = "AI-generated" if paragraph_score >= cutoff else "Human-written"
    confidence_pct = round(paragraph_score * 100, 1)
    print(f"\n{'='*55}")
    print(f"Verdict      : {verdict}")
    print(f"AI confidence: {confidence_pct}%")
    print(f"{'='*55}")
    print("\nSentence breakdown:")
    for item in sentence_scores:
        flag = "  [AI]" if item["ai_probability"] >= cutoff else "      "
        pct = round(item["ai_probability"] * 100, 1)
        print(f"{flag} {pct:5.1f}%  {item['sentence'][:90]}")
    print()
    return {"verdict": verdict, "paragraph_ai_score": paragraph_score, "sentences": sentence_scores}


def load_pipeline(vectorizer_path="tfidf_vectorizer.pkl",
                  xgb_model_path="xgb_model.pkl",
                  xgb_encoder_path="xgb_encoder.pkl",
                  bert_dir=SAVE_DIR,
                  weights_path="ensemble_weights.pkl"):
    vectorizer = load_vectorizer(vectorizer_path)
    xgb_model = joblib.load(xgb_model_path)
    bert_model, tokenizer, _ = load_bert(bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    xgb_w, bert_w = load_ensemble_weights(weights_path)
    return vectorizer, xgb_model, bert_model, tokenizer, device, xgb_w, bert_w


def run(paragraph, vectorizer, xgb_model, bert_model, tokenizer, device, xgb_w, bert_w):
    paragraph_score = score_paragraph(
        paragraph, vectorizer, xgb_model, bert_model, tokenizer, device, xgb_w, bert_w
    )
    sentence_scores = score_sentences(
        paragraph, vectorizer, xgb_model, bert_model, tokenizer, device, xgb_w, bert_w
    )
    report = format_report(paragraph_score, sentence_scores)
    return report


if __name__ == "__main__":
    sample = (
        "The mitochondria are the powerhouse of the cell. They generate ATP through oxidative "
        "phosphorylation and play a central role in apoptosis. Recent studies have demonstrated "
        "that mitochondrial dysfunction is implicated in a wide range of human diseases including "
        "Parkinson's and Alzheimer's. The cell membrane regulates what enters and exits the cell."
    )
    pipeline = load_pipeline()
    result = run(sample, *pipeline)
