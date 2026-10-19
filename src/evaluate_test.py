"""Evaluate the fine-tuned transformer on validation data."""

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "models/distilbert-final"
LABELS = ["negative", "neutral", "positive"]

df = pd.read_csv("data/processed/test.csv")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)
model.eval()

predictions = []

with torch.no_grad():
    for start in range(0, len(df), 8):
        texts = df["text"].iloc[start:start + 8].tolist()
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(device)

        outputs = model(**inputs)
        predicted_ids = outputs.logits.argmax(dim=-1).cpu().tolist()
        predictions.extend(model.config.id2label[i] for i in predicted_ids)

print("\nDistilBERT Final Test Classification Report:")
print(classification_report(df["label"], predictions, labels=LABELS, digits=4))

print("\nConfusion Matrix (rows=true, columns=predicted):")
print(pd.DataFrame(
    confusion_matrix(df["label"], predictions, labels=LABELS),
    index=LABELS,
    columns=LABELS,
))

errors = df.loc[df["label"] != predictions].copy()
errors["predicted"] = [
    predictions[i] for i in errors.index
]

print(f"\nMisclassified examples: {len(errors)}")
print(errors[["text", "label", "predicted"]].head(5).to_string(index=False))

from pathlib import Path

output_path = Path("reports/test_errors.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)

errors.to_csv(output_path, index=False)

print(f"\nSaved {len(errors)} validation errors to {output_path}")
