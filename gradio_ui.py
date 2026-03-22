# module_8_gradio_ui.py
# Gradio web interface for the AI plagiarism detector.
# Exposes a single text input that returns a verdict string and a per-sentence breakdown table.

import gradio as gr
import pandas as pd
from module_7_inference import load_pipeline, run

pipeline = load_pipeline()

def detect(paragraph):
    # Validates input, runs the inference pipeline, and formats outputs for the Gradio UI.
    # Args: paragraph (str) — raw user-submitted text
    # Returns: (verdict_md: str, breakdown: pd.DataFrame)
    #          verdict_md is a Markdown string; breakdown has columns Sentence / AI probability (%) / Flag
    if not paragraph or len(paragraph.strip()) < 20:
        return "Please enter at least a sentence.", pd.DataFrame()
    result = run(paragraph, *pipeline)
    overall = f"**{result['verdict']}** — AI confidence: {round(result['paragraph_ai_score']*100, 1)}%"
    rows = [{"Sentence": item["sentence"],
             "AI probability (%)": round(item["ai_probability"] * 100, 1),
             "Flag": "AI" if item["ai_probability"] >= 0.65 else "Human"}
            for item in result["sentences"]]
    return overall, pd.DataFrame(rows)

with gr.Blocks(title="AI Plagiarism Detector") as app:
    gr.Markdown("## AI Plagiarism Detector\nPaste a paragraph to check if it was written by an AI.")
    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(label="Input paragraph", placeholder="Paste your text here...", lines=8)
            submit_btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=1):
            verdict_output = gr.Markdown(label="Verdict")
    sentence_table = gr.Dataframe(label="Sentence-level breakdown", interactive=False)
    submit_btn.click(fn=detect, inputs=[text_input], outputs=[verdict_output, sentence_table])
    gr.Examples(
        examples=[
            ["The large language model was trained on a diverse corpus of internet text, enabling it to generate coherent and contextually appropriate responses."],
            ["Cripes women in here are brutal. I am someone who has unpredictable cycles and even at my age, IN MY 40S, have accidents."],
            ["At a time when many eateries across Bengaluru were scaling down their menus, the family moved away from LPG six years ago."],
        ],
        inputs=[text_input],
    )

if __name__ == "__main__":
    app.launch(share=True)