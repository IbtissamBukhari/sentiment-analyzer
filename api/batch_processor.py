import pandas as pd

from api.classifier import predict


def process_csv(file):

    df = pd.read_csv(file)

    if "Text" not in df.columns:
        raise ValueError(
            "CSV must contain a column named 'Text'"
        )

    predictions = []

    for text in df["Text"]:

        result = predict(str(text))

        predictions.append(
            result["label"]
        )

    df["prediction"] = predictions

    return df