FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

COPY requirements-api.txt .

RUN pip install --no-cache-dir -r requirements-api.txt
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY src/ ./src/

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
