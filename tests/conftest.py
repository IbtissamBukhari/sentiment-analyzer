"""Test setup: replace RoBERTa with a small keyword stub.

The stub keeps the tests fast and lets CI run without downloading the
500 MB model or installing PyTorch. It returns output in the same shape as
a transformers text-classification pipeline called with top_k=None.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["PRELOAD_MODEL"] = "0"

import pytest  # noqa: E402

from api import classifier  # noqa: E402

POSITIVE = {"great", "love", "excellent", "perfect", "amazing"}
NEGATIVE = {"terrible", "broke", "awful", "late", "refund"}


class StubPipeline:
    def __call__(self, texts, batch_size=None):
        results = []
        for text in texts:
            words = set(text.lower().replace(".", " ").replace(",", " ").split())
            pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
            raw = {"positive": 1 + 3 * pos, "negative": 1 + 3 * neg, "neutral": 2}
            total = sum(raw.values())
            results.append([{"label": k, "score": v / total} for k, v in raw.items()])
        return results


@pytest.fixture(autouse=True)
def stub_model(monkeypatch):
    monkeypatch.setattr(classifier, "_classifier", StubPipeline())
