from api.batch_processor import CSVError, process_csv
from api.classifier import predict, predict_many
from api.explainer import explain_text

import pytest


def test_predict_returns_all_three_scores():
    result = predict("I love it, it is great")
    assert result["label"] == "positive"
    assert set(result["scores"]) == {"negative", "neutral", "positive"}
    assert abs(sum(result["scores"].values()) - 1) < 1e-3


def test_predict_many_matches_single_predictions():
    texts = ["great product", "it broke, terrible", "it is a box"]
    assert [r["label"] for r in predict_many(texts)] == ["positive", "negative", "neutral"]


def test_explanation_is_for_the_predicted_class():
    # Regression test: LIME used to explain the "neutral" class for every input,
    # so clearly positive words were shown with negative impact.
    result = explain_text("The screen is great and I love the battery", num_samples=200)
    assert result["prediction"] == "positive"
    impacts = {item["word"]: item["impact"] for item in result["explanation"]}
    assert impacts["great"] > 0
    assert impacts["love"] > 0


def test_negative_explanation_highlights_negative_words():
    result = explain_text("It arrived late and broke after a day", num_samples=200)
    assert result["prediction"] == "negative"
    impacts = {item["word"]: item["impact"] for item in result["explanation"]}
    assert impacts["broke"] > 0


def test_csv_accepts_common_column_names():
    df = process_csv(b"Review,stars\ngreat stuff,5\nterrible,1\n")
    assert list(df["sentiment"]) == ["positive", "negative"]
    assert "confidence" in df.columns


def test_csv_without_text_column_is_rejected():
    with pytest.raises(CSVError):
        process_csv(b"a,b\n1,2\n")


def test_csv_row_limit(monkeypatch):
    import api.batch_processor as bp

    monkeypatch.setattr(bp, "MAX_ROWS", 2)
    with pytest.raises(CSVError):
        process_csv(b"text\na\nb\nc\n")
