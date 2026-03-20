import re
import numpy as np
import torch
import joblib
from module_1_data_prep import scrub_text
from module_2_features import load_vectorizer, transform_features
from module_5_bert_finetune import load_bert, SAVE_DIR
from module_6_ensemble import get_bert_probs


def split_into_sentences(paragraph):
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [s.strip() for s in parts if len(s.strip()) > 10]


def get_lr_probs(lr_model, vectorizer, texts):
    feats = transform_features(vectorizer, list(texts))
    return lr_model.predict_proba(feats)[:, 0]


def score_sentences(paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w):
    sentences = split_into_sentences(paragraph)
    if not sentences:
        cleaned = scrub_text(paragraph)
        sentences = [paragraph]
        lr_probs = get_lr_probs(lr_model, vectorizer, [cleaned])
        bert_probs = get_bert_probs(bert_model, tokenizer, [cleaned], device)
    else:
        lr_probs = get_lr_probs(lr_model, vectorizer, [scrub_text(s) for s in sentences])
        bert_probs = get_bert_probs(bert_model, tokenizer, [scrub_text(s) for s in sentences], device)

    blended = lr_w * lr_probs + bert_w * bert_probs
    sentence_scores = [{"sentence": s, "ai_probability": round(float(p), 4)}
                       for s, p in zip(sentences, blended)]
    paragraph_score = float(np.mean(blended))
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
    paragraph_score, sentence_scores = score_sentences(
        paragraph, vectorizer, lr_model, bert_model, tokenizer, device, lr_w, bert_w
    )
    return format_report(paragraph_score, sentence_scores)


if __name__ == "__main__":
    samples = {
        "AI sample"    : "It sounds like it should be a winner—a gripping sci-fi crime story about a brilliant medical examiner unraveling sinister murders in space, backed by a stellar cast including Nicole Kidman, Jamie Lee Curtis, Bobby Cannavale, and Simon Baker. But somehow, it misses the mark completely. Instead of delivering suspense or intrigue, it drags along as a bleak, tedious drama centered on the personal issues of thoroughly unlikable characters. And when the story occasionally remembers there are actual crimes to solve, the investigations feel flat, lacking any real tension or excitement.",
        "Human sample" : "It's supposedly about a brilliant medical examiner who solves baffling crimes involving serial predators and murders aboard spaceships, and it stars Nicole Kidman, Jamie Lee Curtis, Bobby Cannavale and Simon Baker: how could it not be good? Pretty easily, as it turns out, because the people behind this dour trainwreck clearly didn't understand the assignment. It's a tedious relationship drama about the interpersonal tribulations of a bunch of unlikeable clods. On the rare occasions that the dopey script remembers that there are crimes afoot, it doesn't imbue the investigations with any tension or intrigue.",
        "News article" : "ISRO’s NavIC constellation, for which it has launched 11 satellites since 2013, is in operational distress. Only three satellites remain capable of providing position, navigation, and timing (PNT) services, leaving the constellation unable to fulfil its purpose of replacing the U.S.’s GPS system over the Indian subcontinent. A PNT constellation requires at least four PNT-capable satellites, and India had only four until ISRO said an atomic clock onboard the IRNSS-1F satellite failed on March 13. The constellation’s first-generation satellites use rubidium atomic clocks made by Swiss company SpectraTime, and which have been dogged by failure. ISRO’s latest attempt to launch a second-generation satellite, NVS-02, was abortive after the machine was left in the wrong orbit. IRNSS-1F, launched in March 2016, completed its 10-year design life just three days before its clock failed. Eight other satellites have either been decommissioned, have failed to reach orbit or have bad clocks. In 2018, ISRO transitioned to using indigenous rubidium atomic clocks, developed by the ISRO-Space Applications Centre. NVS-01, launched in May 2023, was the first to carry the device; all second-generation NVS series satellites will too.",
    }
    pipeline = load_pipeline()
    for label, text in samples.items():
        print(f"\n>>> {label}")
        run(text, *pipeline)