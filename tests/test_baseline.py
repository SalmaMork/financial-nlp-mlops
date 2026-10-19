"""Smoke tests for the trained baseline classifier."""

import pickle
from pathlib import Path

MODEL_PATH = Path("models/baseline.pkl")
VALID_LABELS = {"negative", "neutral", "positive"}


def test_baseline_predictions():
    with MODEL_PATH.open("rb") as file:
        model = pickle.load(file)

    sentences = [
        "The company reported a significant increase in profits.",
        "Revenue remained unchanged compared with last year.",
        "The company recorded a substantial financial loss.",
    ]

    predictions = model.predict(sentences)

    assert len(predictions) == len(sentences)
    assert set(predictions).issubset(VALID_LABELS)


if __name__ == "__main__":
    test_baseline_predictions()
    print("PASS: Baseline prediction checks")
