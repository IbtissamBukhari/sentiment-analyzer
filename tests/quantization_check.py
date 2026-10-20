"""Compare the 8-bit ONNX model with the original PyTorch model.

Run in CI (see .github/workflows/ci.yml). Needs torch and transformers in
addition to the API requirements, and the ONNX files from api.download_model.
Fails if the two models disagree on more than 10% of the sample reviews.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from transformers import pipeline

from api.classifier import LABELS, OnnxSentimentModel, MODEL_DIR

THRESHOLD = 0.90
reviews = json.loads((Path(__file__).parent / "fixtures" / "sample_reviews.json").read_text())

original = pipeline(
    "text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None,
    truncation=True,
)
quantized = OnnxSentimentModel(MODEL_DIR)


def as_matrix(outputs):
    return np.array([[{d["label"].lower(): d["score"] for d in out}[l] for l in LABELS] for out in outputs])


p_orig = as_matrix(original(reviews))
p_quant = as_matrix(quantized(reviews))

agreement = float((p_orig.argmax(1) == p_quant.argmax(1)).mean())
mean_abs_diff = float(np.abs(p_orig - p_quant).mean())

print(f"Reviews compared:            {len(reviews)}")
print(f"Label agreement:             {agreement:.1%}")
print(f"Mean probability difference: {mean_abs_diff:.4f}")
for text, a, b in zip(reviews, p_orig.argmax(1), p_quant.argmax(1)):
    if a != b:
        print(f"  differs: {LABELS[a]} -> {LABELS[b]}: {text}")

print(f"::notice title=Quantized vs original::label agreement {agreement:.1%} on {len(reviews)} reviews, mean probability difference {mean_abs_diff:.4f}")
summary = os.getenv("GITHUB_STEP_SUMMARY")
if summary:
    with open(summary, "a") as f:
        f.write(f"### Quantized vs original model\n\n"
                f"- Reviews compared: {len(reviews)}\n"
                f"- Label agreement: {agreement:.1%}\n"
                f"- Mean probability difference: {mean_abs_diff:.4f}\n")

sys.exit(0 if agreement >= THRESHOLD else 1)
