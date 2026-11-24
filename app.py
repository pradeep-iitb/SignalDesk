"""
Signal Desk demo (Gradio). Paste a paper title + abstract -> top-3 arXiv areas with confidence.

Local:               python app.py              (uses the model saved in ./model by the notebook)
Hugging Face Space:  upload app.py + requirements.txt and set the Space variable
                     MODEL_ID = <your-username>/signal-desk-distilbert   (the Hub repo you pushed to)
"""
import json
import os
import re
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_ID = os.environ.get("MODEL_ID", "model")
if not Path(MODEL_ID).exists() and "/" not in MODEL_ID:
    raise SystemExit("No model found: run signal_desk.ipynb first (it saves ./model), "
                     "or set the MODEL_ID environment variable to your Hugging Face repo id.")

tok = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
if Path(MODEL_ID).exists():
    meta = json.loads((Path(MODEL_ID) / "signal_desk.json").read_text())
else:
    from huggingface_hub import hf_hub_download
    meta = json.loads(Path(hf_hub_download(MODEL_ID, "signal_desk.json")).read_text())
THRESHOLD = meta["abstain_threshold"]
NAMES = {"nlp": "Language (cs.CL)", "vision": "Vision (cs.CV)", "security": "Security (cs.CR)",
         "robotics": "Robotics (cs.RO)", "networking": "Networking (cs.NI)"}


def clean(s: str) -> str:  # same cleaning as the notebook
    s = re.sub(r"\$[^$]*\$", " ", s)
    s = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def classify(text: str):
    if not text or not text.strip():
        return {}, "Paste a title and abstract to sort it."
    enc = tok(clean(text), truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, -1)[0]
    scores = {NAMES.get(model.config.id2label[i], model.config.id2label[i]): float(p) for i, p in enumerate(probs)}
    top = max(scores, key=scores.get)
    if scores[top] < THRESHOLD:
        verdict = (f"**I don't know.** The best guess is {top} at {scores[top]:.0%}, below the "
                   f"{THRESHOLD:.0%} confidence the desk needs to answer. This text may sit between areas.")
    else:
        verdict = f"**{top}**, {scores[top]:.0%} confident."
    return scores, verdict


EXAMPLES = [
    "Jailbreaking aligned chatbots with adversarial suffixes. We show that short optimised token strings "
    "make safety-tuned language models produce harmful content and transfer across models.",
    "A quadruped robot learns to climb stairs from depth images using reinforcement learning in simulation "
    "and transfers to hardware without fine-tuning.",
    "We study congestion control for 5G mobile networks and propose a scheduler that cuts latency for "
    "video streaming users at the cell edge.",
]

with gr.Blocks(title="Signal Desk") as demo:
    gr.Markdown("# Signal Desk\nPaste a computer-science paper's title and abstract. The desk sorts it into "
                "one of five arXiv areas and says how sure it is. Below "
                f"{THRESHOLD:.0%} confidence it answers *I don't know*.")
    with gr.Row():
        with gr.Column(scale=3):
            box = gr.Textbox(label="Title and abstract", lines=9, placeholder="Paste text here")
            btn = gr.Button("Sort it", variant="primary")
        with gr.Column(scale=2):
            label = gr.Label(label="Top 3 areas", num_top_classes=3)
            verdict = gr.Markdown()
    gr.Examples(EXAMPLES, inputs=box)
    gr.Markdown(f"Fine-tuned DistilBERT · macro-F1 {meta['test_macroF1']:.3f} on 2025 papers, "
                f"{meta['shifted_macroF1']:.3f} on 2012-14 papers.")
    btn.click(classify, box, [label, verdict], api_name="classify")
    box.submit(classify, box, [label, verdict])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"))
