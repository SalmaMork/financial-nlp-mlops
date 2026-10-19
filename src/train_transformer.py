"""Fine-tune DistilBERT for financial sentiment classification."""

import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

MODEL_NAME = "distilbert/distilbert-base-uncased"

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def load_split(filename):
    df = pd.read_csv(f"data/processed/{filename}.csv")
    df["label"] = df["label"].map(LABEL2ID)
    return Dataset.from_pandas(df, preserve_index=False)


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

train_dataset = load_split("train")
validation_dataset = load_split("validation")


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )


train_dataset = train_dataset.map(tokenize, batched=True)
validation_dataset = validation_dataset.map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    label2id=LABEL2ID,
    id2label=ID2LABEL,
)

print("Training examples:", len(train_dataset))
print("Validation examples:", len(validation_dataset))
print("MPS available:", torch.backends.mps.is_available())
print("Model and datasets prepared successfully!")


def compute_metrics(prediction):
    predictions = np.argmax(prediction.predictions, axis=-1)
    labels = prediction.label_ids

    return {
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro"),
    }


training_args = TrainingArguments(
    output_dir="models/distilbert-checkpoints",
    num_train_epochs=2,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,
    logging_steps=50,
    seed=42,
    report_to="none",
    dataloader_pin_memory=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("\nTrainer configured successfully!")
print("Epochs:", training_args.num_train_epochs)
print("Training batch size:", training_args.per_device_train_batch_size)
print("Evaluation metric: macro F1")

if __name__ == "__main__":
    print("\nStarting DistilBERT fine-tuning...")
    trainer.train()

    print("\nFinal validation results:")
    print(trainer.evaluate())

    trainer.save_model("models/distilbert-final")
    tokenizer.save_pretrained("models/distilbert-final")

    print("\nFine-tuned model saved to models/distilbert-final")
