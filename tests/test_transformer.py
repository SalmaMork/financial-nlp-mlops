"""Smoke test for the fine-tuned DistilBERT classifier."""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "models/distilbert-final"
VALID_LABELS = {"negative", "neutral", "positive"}


def test_transformer_predictions():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    sentences = [
        "The company reported an increase in profits.",
        "Revenue remained unchanged.",
        "The company recorded a financial loss.",
    ]

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits

    predictions = [
        model.config.id2label[i]
        for i in logits.argmax(dim=-1).tolist()
    ]

    assert logits.shape == (len(sentences), 3)
    assert len(predictions) == len(sentences)
    assert set(predictions).issubset(VALID_LABELS)


if __name__ == "__main__":
    test_transformer_predictions()
    print("PASS: DistilBERT prediction checks")
