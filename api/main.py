import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api import classifier
from api.batch_processor import MAX_ROWS, CSVError, process_csv
from api.explainer import explain_text
from api.schemas import ExplanationResponse, PredictionResponse, TextInput

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model at startup so the first visitor does not wait for it
    if os.getenv("PRELOAD_MODEL", "1") == "1":
        classifier.get_classifier()
    yield


app = FastAPI(
    title="Sentiment Analysis API",
    description="Three-class sentiment analysis with RoBERTa and word-level LIME explanations.",
    version="1.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.api_route("/health", methods=["GET", "HEAD"])
def health():
    return {"status": "ok", "model_loaded": classifier.is_loaded()}


@app.get("/model-info")
def model_info():
    return {
        "model": classifier.MODEL_NAME,
        "labels": classifier.LABELS,
        "evaluation": {
            "dataset": "Amazon Fine Food Reviews, balanced sample of 15,000 (5,000 per class)",
            "accuracy_3_class": 0.61,
            "accuracy_on_positive_and_negative_reviews": 0.814,
            "recall": {"negative": 0.74, "neutral": 0.20, "positive": 0.88},
        },
        "max_csv_rows": MAX_ROWS,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(data: TextInput):
    return classifier.predict(data.text)


@app.post("/explain", response_model=ExplanationResponse)
def explain(data: TextInput):
    return explain_text(data.text)


@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File is larger than 2 MB.")
    try:
        df = process_csv(raw)
    except CSVError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    output = io.BytesIO()
    df.to_csv(output, index=False)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=sentiment_predictions.csv"},
    )
