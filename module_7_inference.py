import re
import numpy as np
import torch
import joblib
from module_1_data_prep import scrub_text, clean_for_bert
from module_2_features import load_vectorizer, transform_features
from module_5_bert_finetune import load_bert, SAVE_DIR
from module_6_ensemble import get_bert_probs


def split_into_sentences(paragraph):
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in parts if len(s.strip()) > 10]


def get_lr_probs(lr_model, vectorizer, texts):
    """Return AI probability from Logistic Regression.
    LabelEncoder: AI=0, Human=1 → predict_proba[:, 0] = P(AI)."""
    feats = transform_features(vectorizer, list(texts))
    return lr_model.predict_proba(feats)[:, 0]


def score_paragraph(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w):
    """Score the full paragraph first, then break down by sentence."""

    # --- Paragraph-level score (primary) ---
    lr_clean = scrub_text(paragraph)
    bert_clean = clean_for_bert(paragraph)

    lr_prob = get_lr_probs(lr_model, vectorizer, [lr_clean])
    bert_prob = get_bert_probs(bert_model, tokenizer, [bert_clean], device)
    paragraph_score = float(lr_w * lr_prob[0] + bert_w * bert_prob[0])

    # --- Sentence-level scores (supplementary detail) ---
    sentences = split_into_sentences(paragraph)
    sentence_scores = []
    if len(sentences) > 1:
        lr_sent_probs = get_lr_probs(lr_model, vectorizer, [scrub_text(s) for s in sentences])
        bert_sent_probs = get_bert_probs(bert_model, tokenizer, [clean_for_bert(s) for s in sentences], device)
        blended = lr_w * lr_sent_probs + bert_w * bert_sent_probs
        sentence_scores = [{"sentence": s, "ai_probability": round(float(p), 4)}
                           for s, p in zip(sentences, blended)]
    else:
        # Only one sentence — reuse the paragraph score
        sentence_scores = [{"sentence": paragraph.strip(), "ai_probability": round(paragraph_score, 4)}]

    return paragraph_score, sentence_scores


def format_report(paragraph_score, sentence_scores):
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


def load_pipeline(vectorizer_path="tfidf_vectorizer.pkl",
                  lr_model_path="lr_model.pkl",
                  bert_dir=SAVE_DIR):
    vectorizer = load_vectorizer(vectorizer_path)
    lr_model = joblib.load(lr_model_path)
    bert_model, tokenizer, _ = load_bert(bert_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_model = bert_model.to(device)
    lr_w  = 0.1
    bert_w = 0.9
    return vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w


def run(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w):
    paragraph_score, sentence_scores = score_paragraph(
        paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w
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