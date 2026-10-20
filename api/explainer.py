"""Word-level explanations with LIME.

LIME perturbs the input by removing words, scores every variant with the
classifier, and fits a simple linear model to estimate how much each word
pushed the prediction towards the predicted class.
"""

import os

import numpy as np
from lime.lime_text import LimeTextExplainer

from api.classifier import LABELS, score_batch

NUM_SAMPLES = int(os.getenv("LIME_NUM_SAMPLES", "300"))

_explainer = LimeTextExplainer(class_names=LABELS, random_state=42)


def _predict_proba(texts):
    return np.array([[s.get(label, 0.0) for label in LABELS] for s in score_batch(texts)])


def explain_text(text: str, num_features: int = 10, num_samples: int = NUM_SAMPLES) -> dict:
    probabilities = _predict_proba([text])[0]
    predicted = int(np.argmax(probabilities))

    # Explain the predicted class. Without `labels`, LIME explains class 1
    # ("neutral") for every input, which made positive words look negative.
    exp = _explainer.explain_instance(
        text_instance=text,
        classifier_fn=_predict_proba,
        num_features=num_features,
        num_samples=num_samples,
        labels=(predicted,),
    )

    words = [
        {"word": str(word), "impact": round(float(weight), 4)}
        for word, weight in exp.as_list(label=predicted)
    ]
    words.sort(key=lambda w: abs(w["impact"]), reverse=True)

    return {
        "prediction": LABELS[predicted],
        "confidence": round(float(probabilities[predicted]), 4),
        "explanation": words,
    }
