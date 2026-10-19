# Financial Sentiment Analysis — DistilBERT Fine-Tuning & MLOps

An end-to-end financial NLP project comparing a traditional machine learning baseline with a fine-tuned transformer, followed by evaluation, API deployment, automated testing, and Docker packaging.

## Project Overview

This project classifies financial news sentences into three sentiment categories:

- **Positive**
- **Neutral**
- **Negative**

It uses the **Financial PhraseBank (AllAgree)** dataset and compares:

1. **Baseline:** TF-IDF features with Logistic Regression
2. **Transformer:** Fine-tuned DistilBERT

The final transformer model is served through a **FastAPI REST API** and packaged with **Docker** for reproducible inference.

## Test Set Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| TF-IDF + Logistic Regression | 90.71% | 0.8837 |
| Fine-tuned DistilBERT | **96.02%** | **0.9451** |

DistilBERT improved test macro F1 by **6.14 percentage points** over the baseline.

Evaluation used a held-out test set of **226 sentences**. These results reflect one stratified train/validation/test split rather than a multi-seed benchmark.

## Technical Stack

- **Language:** Python 3.11
- **Machine learning:** scikit-learn
- **Deep learning:** PyTorch, Hugging Face Transformers
- **NLP model:** DistilBERT
- **API:** FastAPI, Uvicorn
- **Testing:** pytest
- **Deployment:** Docker (CPU inference)

## Key Features

- Stratified data splitting and preprocessing
- Traditional ML baseline for comparison
- DistilBERT fine-tuning and evaluation
- Validation and test error analysis
- REST API with sentiment and confidence predictions
- Automated tests for data, models, and API
- Dockerized inference service

## Run the API with Docker

The fine-tuned model is publicly hosted on Hugging Face:

https://huggingface.co/SalmaMork/financial-sentiment-distilbert

The API automatically downloads the model when it starts.
No local model weights are required to build the Docker image.
An internet connection is needed for the initial download.

### Build the Docker image

    docker build -t financial-nlp-api .

### Start the API

    docker run --rm -p 8000:8000 financial-nlp-api

### API Endpoints

- `GET /health` — Check API status
- `POST /predict` — Predict financial sentiment
- `/docs` — Interactive Swagger documentation

### Example Prediction

Send this JSON to `POST /predict`:

    {"text": "The company reported a significant increase in profits this year."}

Example response:

    {"sentiment": "positive", "confidence": 0.9967}

## Automated Tests

Run the test suite after installing development dependencies
and preparing the dataset and model artifacts:

    python -m pytest tests/ -v
