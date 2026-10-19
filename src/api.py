"""REST API for financial sentiment classification."""

from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "SalmaMork/financial-sentiment-distilbert"
model = None
tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    yield


app = FastAPI(
    title="Financial Sentiment NLP API",
    description="Financial sentiment classification using fine-tuned DistilBERT.",
    version="1.0.0",
    lifespan=lifespan,
)


class PredictionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest):
    inputs = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(probabilities.argmax())

    return {
        "sentiment": model.config.id2label[predicted_id],
        "confidence": round(float(probabilities[predicted_id]), 4),
    }
