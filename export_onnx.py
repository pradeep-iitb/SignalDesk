"""
Export the fine-tuned model in model/ to ONNX for the browser demo (transformers.js layout):

    python export_onnx.py

Writes model/onnx/model.onnx (full precision) and model/onnx/model_quantized.onnx (int8), then
checks on real texts that both give the same top class as the PyTorch model.
Needs: onnx, onnxruntime (see requirements.txt).
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent
MODEL, OUT = ROOT / "model", ROOT / "model" / "onnx"
OUT.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).eval()

enc = tok(["a sample text", "another one that is longer than the first"], padding=True, return_tensors="pt")
torch.onnx.export(model, (enc["input_ids"], enc["attention_mask"]), str(OUT / "model.onnx"),
                  input_names=["input_ids", "attention_mask"], output_names=["logits"],
                  dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                "logits": {0: "batch"}}, opset_version=17, dynamo=False)
quantize_dynamic(str(OUT / "model.onnx"), str(OUT / "model_quantized.onnx"), weight_type=QuantType.QInt8)
for f in ("model.onnx", "model_quantized.onnx"):
    print(f"{f}: {(OUT / f).stat().st_size / 1e6:.1f} MB")

# agreement check: 300 main-set texts + the whole shifted set
df = pd.concat([pd.read_csv(ROOT / "data" / "arxiv_main.csv").sample(300, random_state=0),
                pd.read_csv(ROOT / "data" / "arxiv_shifted.csv")])
texts = (df["title"] + ". " + df["abstract"]).tolist()
sessions = {f: ort.InferenceSession(str(OUT / f), providers=["CPUExecutionProvider"])
            for f in ("model.onnx", "model_quantized.onnx")}
pt, ox = [], {f: [] for f in sessions}
for i in range(0, len(texts), 32):
    e = tok(texts[i:i + 32], truncation=True, max_length=256, padding=True, return_tensors="np")
    with torch.no_grad():
        pt.append(model(input_ids=torch.tensor(e["input_ids"]),
                        attention_mask=torch.tensor(e["attention_mask"])).logits.numpy())
    for f, s in sessions.items():
        ox[f].append(s.run(None, {"input_ids": e["input_ids"].astype(np.int64),
                                  "attention_mask": e["attention_mask"].astype(np.int64)})[0])
pt = np.concatenate(pt)
for f in sessions:
    o = np.concatenate(ox[f])
    print(f"{f}: same top class as PyTorch on {100 * (o.argmax(1) == pt.argmax(1)).mean():.1f} % of {len(pt)} texts")
