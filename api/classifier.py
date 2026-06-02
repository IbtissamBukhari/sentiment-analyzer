from transformers import pipeline

# Load RoBERTa model once (IMPORTANT: not inside function)
classifier = pipeline(
    task="text-classification",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    top_k=None
)


def predict(text: str):
    """
    Returns sentiment prediction from RoBERTa
    """

    result = classifier(text[:512])[0]

    score_dict = {
        item["label"]: item["score"]
        for item in result
    }

    predicted_label = max(score_dict, key=score_dict.get)

    return {
        "label": predicted_label,
        "scores": score_dict
    }