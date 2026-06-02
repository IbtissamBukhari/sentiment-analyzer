import numpy as np
from lime.lime_text import LimeTextExplainer
from api.classifier import classifier


CLASS_NAMES = ["negative", "neutral", "positive"]

explainer = LimeTextExplainer(class_names=CLASS_NAMES)


def predict_proba(texts):
    outputs = []

    for text in texts:

        result = classifier(text[:512])[0]

        score_dict = {
            item["label"]: item["score"]
            for item in result
        }

        outputs.append([
            score_dict["negative"],
            score_dict["neutral"],
            score_dict["positive"]
        ])

    return np.array(outputs)


def explain_text(text: str):

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=10,
        num_samples=500
    )

    explanation = exp.as_list()

    result = {
        "prediction": CLASS_NAMES[np.argmax(exp.predict_proba)],
        "confidence": float(np.max(exp.predict_proba)),
        "explanation": []
    }

    for word, score in explanation:
        result["explanation"].append({
            "word": str(word),
            "impact": float(score)
        })

    result["explanation"] = sorted(
        result["explanation"],
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return result