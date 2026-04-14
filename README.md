🚀 Fake News Detection — MLOps Pipeline

An end-to-end machine learning pipeline for detecting fake news from headlines, built with a production-ready architecture including API deployment, CI/CD, and containerization.

🧠 Overview

This project implements a scalable ML system that:

Cleans and preprocesses raw text data
Trains and evaluates a classification model
Serves predictions via a REST API
Automates workflows using CI/CD
⚙️ Architecture
Input (Headline)
   ↓
TF-IDF Vectorizer
   ↓
Logistic Regression
   ↓
Prediction (Fake / Real)
✨ Features
🔄 End-to-end ML pipeline (data → training → evaluation → inference)
⚡ FastAPI-based REST API for real-time predictions
🐳 Dockerized deployment for portability
🔁 CI/CD with GitHub Actions (training + testing + build)
🧪 Automated testing using pytest
📊 Performance tracking with evaluation reports
🏗️ Project Structure
fake-news-pipeline/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_pipeline.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── api/
│   ├── app.py
│   └── schemas.py
├── models/
├── reports/
├── tests/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml
├── config.yaml
├── Makefile
└── requirements.txt
🚀 Quick Start
1️⃣ Install Dependencies
pip install -r requirements.txt
2️⃣ Run the Pipeline
python -m src.data_pipeline
python -m src.train
python -m src.evaluate

Or:

make all
3️⃣ Make Predictions
CLI
python -m src.predict "Government announces new policy"
API
uvicorn api.app:app --reload

👉 Visit: http://localhost:8000/docs

🔌 API Endpoints
Method	Endpoint	Description
GET	/health	Health check
POST	/predict	Fake/Real prediction
GET	/metrics	Model metrics
Example Request
curl -X POST http://localhost:8000/predict \
-H "Content-Type: application/json" \
-d '{"title": "Breaking: Shocking truth revealed"}'
🧪 Testing
pytest tests/ -v

Or:

make test
🐳 Docker
docker build -t fake-news-api .
docker run -p 8000:8000 fake-news-api

Or:

docker-compose up --build
📊 Model Performance
Metric	Score
Accuracy	0.84
Precision	0.84
Recall	0.84
F1 Score	0.84
🔁 CI/CD Pipeline

Automated using GitHub Actions:

Code linting (flake8)
Data preprocessing
Model training & evaluation
Test execution
Docker image build
Artifact upload
⚙️ Configuration

All settings are defined in config.yaml:

Data paths
Model hyperparameters
API settings
Output/report paths
🎯 Future Improvements
🔐 Authentication & rate limiting
🗄️ Database integration (MongoDB/PostgreSQL)
📈 Model monitoring & drift detection
🤖 Advanced models (BERT / transformers)
💡 Key Highlights
Production-style ML system (not just a model)
Integrated MLOps practices (CI/CD, Docker, testing)
Designed for scalability and deployment
⭐ Star this repo if you found it useful!
