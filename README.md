# Explainable Sentiment Analysis

[![CI](https://github.com/IbtissamBukhari/sentiment-analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/IbtissamBukhari/sentiment-analyzer/actions/workflows/ci.yml)

A web app and REST API that classifies text as positive, neutral or negative with a RoBERTa transformer, and explains each prediction word by word with LIME. It also scores whole CSV files of reviews in one request. The model is quantized to 8-bit and served with ONNX Runtime, so the whole app runs in under 512 MB of memory on free hosting.

**Live demo:** https://sentiment-analyzer-rm0p.onrender.com
**API docs:** https://sentiment-analyzer-rm0p.onrender.com/docs

Hosted on Render's free plan: if the app has been idle, the first request can take up to a minute while it wakes up.

---

## Features

- **Three-class sentiment** from `cardiffnlp/twitter-roberta-base-sentiment-latest`, with the probability of every class
- **Word-level explanations** with LIME: the review is shown with each word highlighted by how strongly it pushed the model towards its prediction
- **Batch analysis**: upload a CSV (up to 500 rows) and get it back with `sentiment` and `confidence` columns
- **FastAPI backend** with input validation, interactive docs at `/docs`, and a single-page frontend served by the same app
- **Optimised for small servers**: uses the 8-bit quantized ONNX export of the model (126 MB instead of 499 MB) on ONNX Runtime, without PyTorch
- **Docker image** with the model baked in, CI that checks the quantized model against the original and load-tests the app inside a 512 MB memory limit

## Results

The model was evaluated in notebook 02 on a balanced sample of 15,000 Amazon Fine Food reviews (5,000 per class). Star ratings were mapped to labels: 1-2 stars negative, 3 stars neutral, 4-5 stars positive.

| Metric | Value |
|:--|:--|
| Accuracy, three classes | 61% |
| Correct on positive and negative reviews | 81% |
| Recall: negative / neutral / positive | 74% / 20% / 88% |

|  | Predicted negative | Predicted neutral | Predicted positive |
|:--|--:|--:|--:|
| **Negative** | 3,723 | 587 | 690 |
| **Neutral** | 1,966 | 993 | 2,041 |
| **Positive** | 254 | 334 | 4,412 |

**What this shows.** The model is reliable at separating positive from negative reviews. Three-star reviews are the weak point: they usually mix praise and complaints ("tastes good, but overpriced"), and the model tends to commit to one side instead of predicting neutral. Part of this is label noise, since a 3-star rating does not always mean neutral text. Fine-tuning on product reviews is the most promising next step (see below).

## How it works

1. **Preprocessing experiment (notebook 01).** Built a spaCy pipeline (lowercasing, lemmatisation, stop-word removal that keeps negations such as "not" and "never") and cleaned the 15,000-review sample.
2. **Model selection (notebook 02).** Compared Cardiff RoBERTa, a multilingual DistilBERT and a DeBERTa aspect-based model on real reviews. RoBERTa was the most accurate. RoBERTa also scored better on raw text than on spaCy-cleaned text, because cleaning removes context the transformer relies on, so the app sends raw text to the model.
3. **Explainability (notebook 03).** LIME removes words at random, re-scores a few hundred variants of the text, and fits a linear model to estimate each word's contribution to the predicted class.
4. **Deployment optimisation.** The full PyTorch model needs well over 512 MB of memory, more than free hosting tiers allow. The app instead uses the community ONNX export of the same model with 8-bit weight quantization ([Xenova/twitter-roberta-base-sentiment-latest](https://huggingface.co/Xenova/twitter-roberta-base-sentiment-latest)), run with ONNX Runtime. CI compares it with the original PyTorch model on 40 sample reviews and fails if the predicted labels agree on less than 90% of them. In the latest run they agreed on 39 of 40 (97.5%), with an average probability difference of 0.026, and the app peaked at about 375 MB under a 512 MB limit while serving a prediction, an explanation and a 200-row CSV.

## API

| Method | Endpoint | Description |
|:--:|:--|:--|
| POST | `/predict` | Sentiment label, confidence and all class probabilities |
| POST | `/explain` | Prediction plus the 10 most influential words (takes 10-30 s) |
| POST | `/predict_csv` | Upload a CSV, download it with predictions added |
| GET | `/model-info` | Model name and evaluation results |
| GET | `/health` | Health check, including whether the model is loaded |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Arrived quickly and works perfectly."}'
```

```json
{
  "label": "positive",
  "confidence": 0.9861,
  "scores": {"negative": 0.0039, "neutral": 0.01, "positive": 0.9861}
}
```

Text must be 1-5,000 characters. For CSV uploads the text column can be called `text`, `review`, `comment`, `content` or `feedback`.

## Run locally

### With Docker (recommended)

```bash
git clone https://github.com/IbtissamBukhari/sentiment-analyzer.git
cd sentiment-analyzer
docker compose up --build
```

Open http://localhost:8000. The container runs with a 512 MB memory limit, the same as the production host.

### With Python

Requires Python 3.11.

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r api/requirements.txt
python -m api.download_model    # fetches the 126 MB model into models/
uvicorn api.main:app --reload
```

Open http://127.0.0.1:8000.

### Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests replace RoBERTa with a small keyword stub, so they run in seconds without downloading the model. CI runs three jobs on every push:

- **test**: the pytest suite
- **quantized vs original model**: label agreement between the 8-bit ONNX model and the original PyTorch model
- **docker (512 MB limit)**: builds the image, runs it with a hard 512 MB memory limit, and sends a prediction, an explanation and a 200-row CSV through it

## Deployment

Deployed on Render's free plan using `render.yaml` (Docker runtime, health check on `/health`). The container reads its port from `$PORT`, so it also runs on other container hosts.

On the free plan the service sleeps when idle, and the first request after a pause can take up to a minute.

## Project structure

```
sentiment-analyzer/
├── api/
│   ├── main.py              FastAPI app and endpoints
│   ├── classifier.py        ONNX model loading and batched inference
│   ├── download_model.py    Fetches the quantized model
│   ├── explainer.py         LIME explanations
│   ├── batch_processor.py   CSV scoring
│   ├── schemas.py           Request and response validation
│   └── requirements.txt
├── frontend/                Single-page app served at /
├── notebooks/               Preprocessing, model comparison, LIME
├── tests/                   Pytest suite, quantization check
├── .github/workflows/       CI pipeline
├── Dockerfile
├── docker-compose.yml
└── render.yaml
```

## Limitations and next steps

- Neutral reviews are often misclassified (20% recall). Fine-tuning RoBERTa on Amazon reviews should help most here.
- The model was trained on tweets, so very long or domain-specific text can be less accurate. Inputs longer than 512 tokens are truncated.
- LIME explanations are approximate and can vary slightly between runs.
- 8-bit quantization can shift probabilities slightly compared with the original model; CI tracks how often labels differ.
- English only.

## Tech stack

Python, RoBERTa, Hugging Face Transformers, ONNX Runtime, model quantization, LIME, spaCy, scikit-learn, Pandas, FastAPI, Pydantic, HTML/CSS/JavaScript, Docker, GitHub Actions, Pytest, Render

## Author

**Syed Ali Ibtissam Bukhari**
[GitHub](https://github.com/IbtissamBukhari) · [LinkedIn](https://www.linkedin.com/in/syed-ali-ibtissam-bukhari-50438028a)

## License

MIT
