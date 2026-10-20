"""Score every row of an uploaded CSV."""

import io
import os

import pandas as pd

from api.classifier import predict_many

MAX_ROWS = int(os.getenv("MAX_CSV_ROWS", "500"))
TEXT_COLUMNS = ("text", "review", "comment", "content", "feedback")


class CSVError(ValueError):
    """Raised for problems the user can fix in their file."""


def find_text_column(df: pd.DataFrame) -> str:
    lookup = {col.strip().lower(): col for col in df.columns}
    for name in TEXT_COLUMNS:
        if name in lookup:
            return lookup[name]
    raise CSVError(
        "CSV must have a text column named one of: " + ", ".join(TEXT_COLUMNS)
    )


def process_csv(raw: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # pandas raises several parser error types
        raise CSVError(f"Could not read the file as CSV: {exc}") from exc

    if df.empty:
        raise CSVError("The CSV has no rows.")
    if len(df) > MAX_ROWS:
        raise CSVError(f"The CSV has {len(df)} rows; the limit is {MAX_ROWS}.")

    column = find_text_column(df)
    texts = df[column].fillna("").astype(str).tolist()

    results = predict_many([t if t.strip() else " " for t in texts])
    df["sentiment"] = [r["label"] for r in results]
    df["confidence"] = [r["confidence"] for r in results]
    return df
