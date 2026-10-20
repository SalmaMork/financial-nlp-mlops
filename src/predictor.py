"""Object-oriented financial sentiment prediction."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SentimentPredictor:
    """Load a sentiment model and make predictions."""

    def __init__(self, model_path="SalmaMork/financial-sentiment-distilbert"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def predict(self, text: str) -> dict:
        """Predict the sentiment of financial text."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
        )

        with torch.inference_mode():
            logits = self.model(**inputs).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        predicted_id = int(probabilities.argmax())

        return {
            "sentiment": self.model.config.id2label[predicted_id],
            "confidence": round(float(probabilities[predicted_id]), 4),
        }
