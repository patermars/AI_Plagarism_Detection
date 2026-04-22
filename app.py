import io
from flask import Flask, request, jsonify, render_template
from inference import load_pipeline, score_paragraph

app = Flask(__name__)
pipeline = load_pipeline()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file provided."}), 400

    filename = f.filename.lower()
    try:
        if filename.endswith(".txt"):
            text = f.read().decode("utf-8", errors="ignore")

        elif filename.endswith(".pdf"):
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(f.read()))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)

        elif filename.endswith(".docx"):
            import docx
            doc = docx.Document(io.BytesIO(f.read()))
            text = "\n".join(p.text for p in doc.paragraphs)

        else:
            return jsonify({"error": "Unsupported file type. Use .txt, .pdf, or .docx"}), 400
    except Exception as e:
        return jsonify({"error": f"Could not read file: {e}"}), 400

    text = text.strip()
    if len(text) < 20:
        return jsonify({"error": "File appears to be empty or too short."}), 400

    return jsonify({"text": text})


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = (data or {}).get("text", "").strip()
    if len(text) < 20:
        return jsonify({"error": "Please enter at least a sentence."}), 400

    bert_model, tokenizer, device = pipeline
    paragraph_score, sentence_scores = score_paragraph(text, bert_model, tokenizer, device)

    ai_count = sum(1 for s in sentence_scores if s["ai_probability"] >= 0.65)
    total     = len(sentence_scores)

    if paragraph_score >= 0.85:
        verdict = "Very likely AI-generated"
    elif paragraph_score >= 0.65:
        verdict = "Possibly AI-generated"
    elif paragraph_score >= 0.40:
        verdict = "Uncertain"
    elif ai_count > 0 and total > 1:
        verdict = f"Likely human-written ({ai_count} AI sentence{'s' if ai_count > 1 else ''} flagged)"
    else:
        verdict = "Likely Human-written"

    return jsonify({
        "verdict": verdict,
        "ai_score": round(float(paragraph_score) * 100, 1),
        "sentences": sentence_scores,
    })

if __name__ == "__main__":
    app.run(debug=False)
