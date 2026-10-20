from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health_and_frontend():
    assert client.get("/health").json()["status"] == "ok"
    page = client.get("/")
    assert page.status_code == 200
    assert "Sentiment Analysis AI" in page.text


def test_predict_endpoint():
    body = client.post("/predict", json={"text": "This is great"}).json()
    assert body["label"] == "positive"
    assert 0 <= body["confidence"] <= 1


def test_blank_and_oversized_text_rejected():
    assert client.post("/predict", json={"text": "   "}).status_code == 422
    assert client.post("/predict", json={"text": "a" * 5001}).status_code == 422


def test_explain_endpoint():
    body = client.post("/explain", json={"text": "I love this, it is excellent"}).json()
    assert body["prediction"] == "positive"
    assert body["explanation"]


def test_csv_endpoint_returns_csv():
    files = {"file": ("reviews.csv", b"text\ngreat\nterrible\n", "text/csv")}
    response = client.post("/predict_csv", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert response.text.splitlines()[0] == "text,sentiment,confidence"


def test_bad_csv_gives_clear_error():
    files = {"file": ("bad.csv", b"a,b\n1,2\n", "text/csv")}
    response = client.post("/predict_csv", files=files)
    assert response.status_code == 400
    assert "text column" in response.json()["detail"]
