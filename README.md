# The Signal Desk

A classifier that reads the title and abstract of a computer-science paper and sorts it into one of
five research areas. Then it is stress-tested on papers from ten years earlier to see how far it can
be trusted when the data changes.

* **Live demo:** https://huggingface.co/spaces/pradeepkumawat/signal-desk
* **Model:** https://huggingface.co/pradeepkumawat/signal-desk-distilbert

| class | arXiv primary category |
|---|---|
| `nlp` | cs.CL: Computation and Language |
| `vision` | cs.CV: Computer Vision and Pattern Recognition |
| `security` | cs.CR: Cryptography and Security |
| `robotics` | cs.RO: Robotics |
| `networking` | cs.NI: Networking and Internet Architecture |

## Data

Collected with `collect_data.py` from the arXiv API.

| set | what | texts | used for |
|---|---|---|---|
| main | papers submitted in **2025**, 520 per class, spread over the four quarters | 2,600 | train 1,820 / validation 390 / test 390 (stratified, seed 42) |
| shifted | papers submitted in **2012-2014**, 60 per class | 299 (1 duplicate removed) | stress test only |

* **Source:** the [arXiv API](https://info.arxiv.org/help/api/index.html), queried by category and
  submission date. Requests are spaced 3 seconds apart, as the
  [API terms of use](https://info.arxiv.org/help/api/tou.html) ask. arXiv metadata is CC0.
* **Label:** the paper's *primary* category, chosen by its authors. Papers that are only
  cross-listed in a category are dropped, so each paper has exactly one label.
* **Text:** title plus abstract.
* **Fields kept:** arXiv id, date, title, abstract and category. No author names or other personal
  data are stored.
* **Cleaning:** done in the notebook. LaTeX math and commands, URLs and e-mail-like strings are
  stripped. Duplicates are removed by id, normalised text and normalised title.

## Results

All three models are trained on the same split. The best settings for each are picked on
validation.

| model | val macro-F1 | **test macro-F1 (2025)** | **shifted macro-F1 (2012-14)** | change |
|---|---|---|---|---|
| A. TF-IDF (1-2 grams) + logistic regression | 0.934 | 0.915 | 0.896 | −1.9 points |
| B. BiLSTM (PyTorch, trained from scratch) | 0.882 | 0.877 | 0.823 | **−5.5 points** |
| C. DistilBERT, fine-tuned (3 epochs) | 0.924 | 0.913 | **0.947** | +3.4 points |

**Which model dropped the most and why.**
- The BiLSTM dropped the most, 5.5 F1 points.
- It isn't because older papers use unseen words. Its out-of-vocabulary rate is about the same on
  both sets: 6.5 % on the 2025 test and 6.0 % on the 2012-14 set.
- It learned word meanings from only 1,820 abstracts with no pre-training, so it relies on
  2025-specific wording. TF-IDF needs only a few stable keywords (*robot*, *attack*, *wireless*),
  and DistilBERT brings general English knowledge from pre-training.
- DistilBERT even improved. The 2012-14 set has fewer boundary-crossing papers: 20.5 % of 2025
  non-NLP papers discuss LLMs, against 0 % in 2012-14.

**Best model.** On the 2025 test set, TF-IDF and DistilBERT are tied. The difference is +0.002,
with a 95 % paired-bootstrap interval of −0.030 to +0.035. DistilBERT is clearly better on the
shifted set, so it is the deployed model.

**Error analysis.** The notebook shows the 10 most confident test mistakes, each with a one-line
reason. All ten are papers that sit between two areas, and 7 of the 10 involve `security`. One
example is an LLM-watermarking paper filed under security. Many are arguably label ambiguity
rather than model errors.

**Bonus: "I don't know".** The desk abstains when its top confidence is below a threshold. The
threshold is set on validation so that the desk answers 90 % of validation texts.

| model (shifted set) | threshold | precision, always answering | precision, with abstain | texts answered |
|---|---|---|---|---|
| A. TF-IDF + LogReg | 0.598 | 0.896 | **0.958** | 79.9 % |
| B. BiLSTM | 0.805 | 0.823 | **0.864** | 88.6 % |
| C. DistilBERT | 0.901 | 0.946 | **0.964** | 92.3 % |

Abstaining raises precision on the shifted set for all three models.

## Demo

The Space is a static page (`space.html`). Hugging Face's free tier hosts only static Spaces;
Gradio Spaces need a paid plan.
- The page runs the fine-tuned DistilBERT in the browser with transformers.js, using an int8 ONNX
  export of the model.
- It shows the top 3 areas with confidence, and answers "I don't know" below the validation
  threshold.
- The int8 model gives the same top answer as the full model on 99.2 % of 600 checked texts.

`app.py` is the same demo as a Gradio app. It runs locally, and it also runs as a Gradio Space
with the Space variable `MODEL_ID` set to the model repo.

## Files

```
├── collect_data.py      data collection (arXiv API)
├── data/                arxiv_main.csv, arxiv_shifted.csv
├── signal_desk.ipynb    cleaning, split, 3 models, stress test, error analysis, bonus, predict(), Hub upload
├── export_onnx.py       ONNX + int8 export of the fine-tuned model for the browser demo
├── app.py               Gradio demo
├── space.html           the Hugging Face Space page
└── requirements.txt
```

The fine-tuned weights (255 MB) are on the Hugging Face Hub, not in this repository. The notebook
writes them to `model/`.

## How to run

```bash
pip install -r requirements.txt      # use a CUDA build of torch for GPU training
python collect_data.py               # optional: re-collects data/ (about 2 minutes)
jupyter nbconvert --to notebook --execute signal_desk.ipynb   # or Run All (about 3 minutes on an RTX 3050 laptop GPU; also runs on CPU or Colab)
python app.py                        # Gradio demo at http://127.0.0.1:7860
python export_onnx.py                # ONNX files for the browser demo -> model/onnx/
```

* **Re-collecting data:** arXiv's listings change over time, so re-collecting gives slightly
  different papers. The committed CSVs are the ones behind the numbers above.
* **Hub upload:** the notebook's last cell uploads `model/` to the Hub when `PUSH_TO_HUB = True`.
  If `model/onnx/` exists, it is uploaded too. The cell reads the access token from the `HF_TOKEN`
  environment variable and the repo name from `HF_REPO_ID`.

## Sources

* arXiv API, https://info.arxiv.org/help/api/index.html. Metadata is CC0. Thank you to arXiv for use
  of its open access interoperability.
* DistilBERT, `distilbert-base-uncased` (Sanh et al., 2019), https://huggingface.co/distilbert-base-uncased
  (Apache 2.0).
* transformers.js 3.8.1 (Apache 2.0), https://huggingface.co/docs/transformers.js
