"""Tests for financial sentiment data preparation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from prepare_data import load_financial_phrasebank


def test_dataset_quality():
    df = load_financial_phrasebank()

    assert len(df) == 2259
    assert df["text"].duplicated().sum() == 0
    assert set(df["label"]) == {"negative", "neutral", "positive"}
    assert df["text"].notna().all()
    assert df["label"].notna().all()


if __name__ == "__main__":
    test_dataset_quality()
    print("PASS: Dataset quality checks")
