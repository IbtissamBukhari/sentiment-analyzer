"""RoBERTa sentiment classifier served with ONNX Runtime.

The model is cardiffnlp/twitter-roberta-base-sentiment-latest, exported to
ONNX and quantized to 8-bit integers (126 MB instead of 499 MB). Running it
with ONNX Runtime instead of PyTorch keeps the whole app inside 512 MB of
memory, so it fits on free hosting tiers.

The model is loaded once, on first use, and every call goes through
score_batch so that the API, LIME and CSV processing share batched inference.
"""

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest (ONNX, 8-bit quantized)"
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
LABELS = ["negative", "neutral", "positive"]
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
MAX_TOKENS = 512

_classifier = None
# One inference at a time: parallel calls on a small CPU instance only slow each other down
_lock = threading.Lock()


class OnnxSentimentModel:
    def __init__(self, model_dir: Path):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        options = ort.SessionOptions()
        options.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
        self.session = ort.InferenceSession(
            str(model_dir / "model_quantized.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = {i.name for i in self.session.get_inputs()}

        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_TOKENS)
        self.tokenizer.enable_padding(pad_id=1, pad_token="<pad>")

        config = json.loads((model_dir / "config.json").read_text())
        self.id2label = {int(k): v.lower() for k, v in config["id2label"].items()}

    def __call__(self, texts: List[str], batch_size: int = BATCH_SIZE) -> List[List[dict]]:
        results = []
        for start in range(0, len(texts), batch_size):
            encodings = self.tokenizer.encode_batch(texts[start:start + batch_size])
            feeds = {
                "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            }
            feeds = {name: value for name, value in feeds.items() if name in self.input_names}
            logits = self.session.run(None, feeds)[0]
            probs = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs /= probs.sum(axis=1, keepdims=True)
            for row in probs:
                results.append([
                    {"label": self.id2label[i], "score": float(p)} for i, p in enumerate(row)
                ])
        return results


def get_classifier():
    global _classifier
    if _classifier is not None:
        return _classifier
    with _lock:
        if _classifier is None:
            _classifier = OnnxSentimentModel(MODEL_DIR)
        return _classifier


def is_loaded() -> bool:
    return _classifier is not None


def score_batch(texts: Sequence[str]) -> List[Dict[str, float]]:
    """Return a {label: probability} dict for each text."""
    if not texts:
        return []
    classifier = get_classifier()
    with _lock:
        outputs = classifier(list(texts), batch_size=BATCH_SIZE)
    return [{item["label"].lower(): float(item["score"]) for item in out} for out in outputs]


def _summarise(scores: Dict[str, float]) -> dict:
    label = max(LABELS, key=lambda l: scores.get(l, 0.0))
    return {
        "label": label,
        "confidence": round(scores[label], 4),
        "scores": {l: round(scores.get(l, 0.0), 4) for l in LABELS},
    }


def predict(text: str) -> dict:
    return _summarise(score_batch([text])[0])


def predict_many(texts: Sequence[str]) -> List[dict]:
    return [_summarise(s) for s in score_batch(texts)]
