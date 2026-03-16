import gradio as gr
import pandas as pd
from module_7_inference import load_pipeline, run


pipeline = load_pipeline()


def detect(paragraph):
    if not paragraph or len(paragraph.strip()) < 20:
        return "Please enter at least a sentence.", pd.DataFrame()

    result = run(paragraph, *pipeline)

    overall = f"**{result['verdict']}** — AI confidence: {round(result['paragraph_ai_score']*100, 1)}%"

    rows = []
    for item in result["sentences"]:
        rows.append({
            "Sentence": item["sentence"],
            "AI probability (%)": round(item["ai_probability"] * 100, 1),
            "Flag": "AI" if item["ai_probability"] >= 0.5 else "Human",
        })
    breakdown = pd.DataFrame(rows)

    return overall, breakdown


with gr.Blocks(title="AI Plagiarism Detector") as app:
    gr.Markdown("## AI Plagiarism Detector\nPaste a paragraph below to check if it was written by an AI.")

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input paragraph",
                placeholder="Paste your text here...",
                lines=8,
            )
            submit_btn = gr.Button("Analyse", variant="primary")
        with gr.Column(scale=1):
            verdict_output = gr.Markdown(label="Verdict")

    sentence_table = gr.Dataframe(
        label="Sentence-level breakdown",
        headers=["Sentence", "AI probability (%)", "Flag"],
        interactive=False,
    )

    submit_btn.click(
        fn=detect,
        inputs=[text_input],
        outputs=[verdict_output, sentence_table],
    )

    gr.Examples(
        examples=[
            ["The large language model was trained on a diverse corpus of internet text, enabling it to generate coherent and contextually appropriate responses across a wide range of domains."],
            ["I honestly think the way the sunset looked yesterday was something I'll never forget. It felt like the sky was on fire, and I just stood there for a while."],
        ],
        inputs=[text_input],
    )


if __name__ == "__main__":
    app.launch(share=False)
