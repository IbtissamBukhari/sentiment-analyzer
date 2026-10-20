from typing import Dict, List, Literal

from pydantic import BaseModel, Field, field_validator

MAX_TEXT_LENGTH = 5000

Sentiment = Literal["negative", "neutral", "positive"]


class TextInput(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT_LENGTH, description="Text to analyse")

    @field_validator("text")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be empty")
        return value.strip()


class PredictionResponse(BaseModel):
    label: Sentiment
    confidence: float
    scores: Dict[str, float]


class ExplanationItem(BaseModel):
    word: str
    impact: float


class ExplanationResponse(BaseModel):
    prediction: Sentiment
    confidence: float
    explanation: List[ExplanationItem]
