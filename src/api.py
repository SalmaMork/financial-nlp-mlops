"""REST API for financial sentiment classification."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

try:
    from src.predictor import SentimentPredictor
except ModuleNotFoundError:
    from predictor import SentimentPredictor


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the sentiment model when the API starts."""
    app.state.predictor = SentimentPredictor()
    yield
    del app.state.predictor


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
def predict(payload: PredictionRequest, request: Request):
    return request.app.state.predictor.predict(payload.text)
