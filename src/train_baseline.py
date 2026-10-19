"""Train and evaluate a TF-IDF + Logistic Regression baseline."""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

train = pd.read_csv("data/processed/train.csv")
validation = pd.read_csv("data/processed/validation.csv")

model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(train["text"], train["label"])

predictions = model.predict(validation["text"])

print("TF-IDF + Logistic Regression Baseline")
print("------------------------------------")
print(f"Validation accuracy: {accuracy_score(validation['label'], predictions):.4f}")
print(f"Validation macro F1: {f1_score(validation['label'], predictions, average='macro'):.4f}")
print("\nClassification report:")
print(classification_report(validation["label"], predictions, digits=4))


# Save the complete TF-IDF + Logistic Regression pipeline.
import pickle
from pathlib import Path

output_path = Path("models/baseline.pkl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("wb") as file:
    pickle.dump(model, file)

print(f"\nBaseline model saved to: {output_path}")
