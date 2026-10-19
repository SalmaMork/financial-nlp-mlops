"""Load and inspect the Financial PhraseBank dataset."""

import zipfile
from collections import Counter

import pandas as pd

ZIP_PATH = "data/FinancialPhraseBank-v1.0.zip"
DATASET_PATH = "FinancialPhraseBank-v1.0/Sentences_AllAgree.txt"


def load_financial_phrasebank():
    with zipfile.ZipFile(ZIP_PATH) as archive:
        text = archive.read(DATASET_PATH).decode("iso-8859-1")

    records = []

    for line in text.splitlines():
        if not line.strip():
            continue

        sentence, label = line.rsplit("@", 1)

        records.append({
            "text": sentence.strip(),
            "label": label.strip(),
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    df = load_financial_phrasebank()

    print("Total examples:", len(df))
    print("Sentiment distribution:", dict(Counter(df["label"])))
    print("Duplicate sentences:", df["text"].duplicated().sum())
    print("\nFirst 3 examples:")
    print(df.head(3).to_string(index=False))


def create_splits(df):
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df["label"],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        stratify=temp_df["label"],
    )

    return train_df, val_df, test_df


if __name__ == "__main__":
    train_df, val_df, test_df = create_splits(load_financial_phrasebank())

    for name, split in [
        ("Train", train_df),
        ("Validation", val_df),
        ("Test", test_df),
    ]:
        print(f"{name}: {len(split)} examples")
        print(split["label"].value_counts().to_dict())


if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)

    print("\nDataset splits saved successfully!")
    for filename in ["train.csv", "validation.csv", "test.csv"]:
        print(output_dir / filename)
