# module_7_inference.py
# Inference pipeline: scores a paragraph (and its individual sentences) for AI authorship
# using the fine-tuned DistilBERT classifier, then formats a human-readable verdict report.

import re
import numpy as np
import torch
from module_5_bert_finetune import load_bert, SAVE_DIR


def split_into_sentences(paragraph):
    # Splits a paragraph into individual sentences on terminal punctuation boundaries.
    # Filters out fragments shorter than 10 characters to avoid noise.
    # Args: paragraph (str)
    # Returns: list[str] — cleaned sentence strings
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in parts if len(s.strip()) > 10]


def get_bert_probs(bert_model, tokenizer, texts, device, batch_size=32, max_len=256):
    # Runs batched forward passes through DistilBERT and returns the softmax probability
    # for the AI class (index 0) for each input text.
    # Args: bert_model — DistilBertForSequenceClassification; tokenizer; texts — iterable of str;
    #       device — torch.device; batch_size (int); max_len (int) — token truncation limit
    # Returns: np.ndarray of shape (N,), dtype float32 — AI probability per text
    bert_model.eval()
    all_probs = []
    text_list = list(texts)
    for start in range(0, len(text_list), batch_size):
        batch_texts = text_list[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_len,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        with torch.no_grad():
            output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(output.logits, dim=-1)[:, 0].cpu().numpy()
        all_probs.extend(probs)
    return np.array(all_probs, dtype=np.float32)


def score_paragraph(paragraph, bert_model, tokenizer, device):
    # Scores the full paragraph as a single unit, then scores each sentence individually.
    # Falls back to the paragraph score when fewer than 2 sentences are detected.
    # Args: paragraph (str); bert_model; tokenizer; device — torch.device
    # Returns: (paragraph_score: float, sentence_scores: list[dict])
    #          sentence_scores dicts contain keys 'sentence' and 'ai_probability'
    paragraph_prob = get_bert_probs(bert_model, tokenizer, [paragraph], device)
    paragraph_score = float(paragraph_prob[0])

    sentences = split_into_sentences(paragraph)
    sentence_scores = []
    if len(sentences) > 1:
        sent_probs = get_bert_probs(bert_model, tokenizer, sentences, device)
        sentence_scores = [
            {"sentence": s, "ai_probability": round(float(p), 4)}
            for s, p in zip(sentences, sent_probs)
        ]
    else:
        sentence_scores = [
            {"sentence": paragraph.strip(), "ai_probability": round(paragraph_score, 4)}
        ]

    return paragraph_score, sentence_scores


def format_report(paragraph_score, sentence_scores):
    # Maps the paragraph AI probability to a verdict label, prints a formatted report,
    # and returns a structured result dict for programmatic consumption.
    # Args: paragraph_score (float) — AI probability [0, 1];
    #       sentence_scores (list[dict]) — per-sentence scores from score_paragraph()
    # Returns: dict with keys 'verdict', 'paragraph_ai_score', 'sentences'
    if paragraph_score >= 0.85:
        verdict = "Very likely AI-generated"
    elif paragraph_score >= 0.65:
        verdict = "Possibly AI-generated"
    elif paragraph_score >= 0.40:
        verdict = "Uncertain"
    else:
        verdict = "Likely Human-written"

    print(f"\n{'='*55}")
    print(f"Verdict      : {verdict}")
    print(f"AI confidence: {round(paragraph_score * 100, 1)}%")
    print(f"{'='*55}")
    print("\nSentence breakdown:")
    for item in sentence_scores:
        flag = "  [AI]" if item["ai_probability"] >= 0.65 else "      "
        print(f"{flag} {round(item['ai_probability']*100,1):5.1f}%  {item['sentence'][:90]}")
    print()
    return {"verdict": verdict, "paragraph_ai_score": paragraph_score, "sentences": sentence_scores}


def load_pipeline(bert_dir=SAVE_DIR):
    # Loads the fine-tuned DistilBERT model and moves it to the best available device.
    # Args: bert_dir (str) — directory containing saved model artefacts
    # Returns: (bert_model, tokenizer, device)
    bert_model, tokenizer, _ = load_bert(bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    return bert_model, tokenizer, device


def run(paragraph, bert_model, tokenizer, device):
    # End-to-end entry point: scores a paragraph and returns the formatted verdict dict.
    # Args: paragraph (str); bert_model; tokenizer; device — torch.device
    # Returns: dict — same structure as format_report()
    paragraph_score, sentence_scores = score_paragraph(
        paragraph, bert_model, tokenizer, device
    )
    return format_report(paragraph_score, sentence_scores)


if __name__ == "__main__":
    samples = {
        "AI sample (ChatGPT)": (
            "It sounds like it should be a winner — a gripping sci-fi crime story about a "
            "brilliant medical examiner unraveling sinister murders in space, backed by a "
            "stellar cast including Nicole Kidman, Jamie Lee Curtis, Bobby Cannavale, and "
            "Simon Baker. But somehow, it misses the mark completely. Instead of delivering "
            "suspense or intrigue, it drags along as a bleak, tedious drama centered on the "
            "personal issues of thoroughly unlikable characters."
        ),
        "Human sample (natural voice)": (
            "It's supposedly about a brilliant medical examiner who solves baffling crimes "
            "involving serial predators and murders aboard spaceships, and it stars Nicole "
            "Kidman, Jamie Lee Curtis, Bobby Cannavale and Simon Baker: how could it not be "
            "good? Pretty easily, as it turns out, because the people behind this dour "
            "trainwreck clearly didn't understand the assignment."
        ),
        "News article (The Hindu / ISRO)": (
            "ISRO's NavIC constellation, for which it has launched 11 satellites since 2013, "
            "is in operational distress. Only three satellites remain capable of providing "
            "position, navigation, and timing (PNT) services, leaving the constellation "
            "unable to fulfil its purpose of replacing the U.S.'s GPS system over the Indian "
            "subcontinent. A PNT constellation requires at least four PNT-capable satellites, "
            "and India had only four until ISRO said an atomic clock onboard the IRNSS-1F "
            "satellite failed on March 13."
        ),
        "Reddit post (informal)": (
            "Cripes women in here are brutal. I am someone who has unpredictable cycles and "
            "even at my age, IN MY 40S, have accidents. It doesn't matter how careful you "
            "are, sometimes it just happens. But nah, let's shame people for bodily functions "
            "they can't control, real mature."
        ),
    }
    pipeline = load_pipeline()
    for label, text in samples.items():
        print(f"\n>>> {label}")
        run(text, *pipeline)