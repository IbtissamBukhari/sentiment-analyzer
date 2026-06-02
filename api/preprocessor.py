import re
import spacy

nlp = spacy.load("en_core_web_sm")

SENTIMENT_STOP_EXCEPTIONS = {
    "never", "no", "not", "neither", "nor",
    "nobody", "nothing", "nowhere", "hardly",
    "barely", "scarcely"
}


def preprocess_text(text: str) -> str:
    """
    Clean and normalize text for sentiment analysis
    (used mainly for logging / optional preprocessing)
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs

    doc = nlp(text)

    tokens = [
        token.lemma_
        for token in doc
        if (not token.is_stop or token.lemma_ in SENTIMENT_STOP_EXCEPTIONS)
        and not token.is_punct
        and not token.is_space
    ]

    return " ".join(tokens)