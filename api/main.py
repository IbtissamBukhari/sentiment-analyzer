from fastapi import FastAPI
from api.schemas import TextInput, PredictionResponse, ExplanationResponse
from api.classifier import predict
from api.explainer import explain_text
from fastapi.middleware.cors import CORSMiddleware
from fastapi import UploadFile, File
from fastapi.responses import StreamingResponse

from api.batch_processor import process_csv

import io


app = FastAPI(
    title="Sentiment Analysis API",
    description="RoBERTa + LIME based sentiment analysis with explainability",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Health Check Endpoint
# ----------------------------
@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API is running"
    }


# ----------------------------
# Prediction Endpoint (Fast)
# ----------------------------
@app.post("/predict", response_model=PredictionResponse)
def get_prediction(data: TextInput):

    result = predict(data.text)

    return {
        "label": result["label"],
        "scores": result["scores"]
    }


# ----------------------------
# Explanation Endpoint (Slow - LIME)
# ----------------------------
@app.post("/explain", response_model=ExplanationResponse)
def get_explanation(data: TextInput):

    result = explain_text(data.text)

    return {
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "explanation": result["explanation"]
    }
@app.post("/predict_csv")
async def predict_csv(
    file: UploadFile = File(...)
):

    df = process_csv(file.file)

    output = io.BytesIO()

    df.to_csv(
        output,
        index=False
    )

    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={
            "Content-Disposition":
            "attachment; filename=predictions.csv"
        }
    )