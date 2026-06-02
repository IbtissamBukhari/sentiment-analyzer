from pydantic import BaseModel


class TextInput(BaseModel):
    text: str


class PredictionResponse(BaseModel):
    label: str
    scores: dict


class ExplanationItem(BaseModel):
    word: str
    impact: float


class ExplanationResponse(BaseModel):
    prediction: str
    confidence: float
    explanation: list[ExplanationItem]