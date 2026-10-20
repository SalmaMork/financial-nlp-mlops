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

## Dataset Preparation

This project uses the **Financial PhraseBank (AllAgree)** dataset.

1. Obtain the original `FinancialPhraseBank-v1.0.zip` archive.
2. Place it in `data/FinancialPhraseBank-v1.0.zip`.
3. From the project root, run:

    python src/prepare_data.py

The script removes duplicate sentences and creates stratified
training (80%), validation (10%), and test (10%) splits using
random seed 42.

The resulting CSV files are saved in `data/processed/`.

Dataset files are excluded from Git to keep the repository lightweight.

## Software Engineering: OOP and Continuous Integration

### Object-Oriented Model Inference

The FastAPI service uses a reusable `SentimentPredictor` class in
`src/predictor.py` to organize model loading, tokenization, and inference.

- Loads the fine-tuned DistilBERT model from Hugging Face.
- Uses PyTorch inference mode for predictions.
- Returns the predicted sentiment and confidence score.
- Separates machine learning inference logic from API routing.

### Continuous Integration (CI)

GitHub Actions automatically runs API integration tests on every push
and pull request.

The workflow is defined in `.github/workflows/tests.yml`.

The automated tests verify:
- The `/health` endpoint returns a successful response.
- The `/predict` endpoint returns a valid sentiment and confidence score.

Run the CI tests locally:

```bash
python -m pytest tests/test_api.py -q


```

Additional tests for data preparation, the baseline model, and the fine-tuned transformer require local datasets or model files.
