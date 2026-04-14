# Fake News Detection — MLOps Pipeline

End-to-end machine learning pipeline for detecting fake news from headlines.

## Architecture

```
Input (headline) → TF-IDF Vectorizer → Logistic Regression → Fake/Real
```

## Project Structure

```
fake-news-pipeline/
├── data/raw/               # Raw datasets
├── data/processed/         # Cleaned data
├── src/                    # ML pipeline source code
│   ├── data_pipeline.py    # Data loading & cleaning
│   ├── train.py            # Model training
│   ├── evaluate.py         # Evaluation & reporting
│   └── predict.py          # CLI prediction tool
├── api/                    # FastAPI REST API
│   ├── app.py              # API endpoints
│   └── schemas.py          # Request/response models
├── models/                 # Saved model artifacts
├── reports/                # Evaluation reports & plots
├── tests/                  # pytest test suite
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile              # Container config
├── docker-compose.yml      # Docker orchestration
├── config.yaml             # Pipeline configuration
├── Makefile                # Convenience commands
└── requirements.txt        # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

```bash
# Step by step:
python -m src.data_pipeline    # Clean raw data
python -m src.train            # Train model
python -m src.evaluate         # Evaluate & generate reports

# Or all at once (requires Make):
make all
```

### 3. Make Predictions

**CLI:**
```bash
python -m src.predict "Government announces new policy on healthcare"
```

**API:**
```bash
# Start the server
uvicorn api.app:app --reload

# Or with Make:
make serve
```

Then visit `http://localhost:8000/docs` for the interactive Swagger UI.

**API Endpoints:**
| Method | Endpoint   | Description                    |
|--------|-----------|--------------------------------|
| GET    | /health   | Health check                   |
| POST   | /predict  | Predict fake/real news         |
| GET    | /metrics  | Latest model metrics           |

**Example API call:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"title": "Breaking: Shocking truth revealed by scientists"}'
```

### 4. Run Tests

```bash
pytest tests/ -v

# Or:
make test
```

### 5. Docker

```bash
# Build & run
docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api

# Or with docker-compose:
docker-compose up --build
```

## Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~0.84  |
| Precision | ~0.84  |
| Recall    | ~0.84  |
| F1 Score  | ~0.84  |

## CI/CD

GitHub Actions pipeline (`.github/workflows/ci.yml`):
1. Lint code with flake8
2. Run data pipeline
3. Train model
4. Evaluate model
5. Run tests
6. Build Docker image
7. Upload model & report artifacts

## Configuration

All settings are in `config.yaml`:
- Data paths
- Model hyperparameters
- API settings
- Report output paths
