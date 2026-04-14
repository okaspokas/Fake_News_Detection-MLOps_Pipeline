.PHONY: install data train evaluate serve test docker-build docker-run all clean

install:
	pip install -r requirements.txt

data:
	python -m src.data_pipeline

train:
	python -m src.train

evaluate:
	python -m src.evaluate

serve:
	uvicorn api.app:app --reload --host 0.0.0.0 --port 8000

test:
	pytest tests/ -v

lint:
	flake8 src/ api/ tests/ --max-line-length=100 --ignore=E501,W503

docker-build:
	docker build -t fake-news-api .

docker-run:
	docker run -p 8000:8000 fake-news-api

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

all: data train evaluate
	@echo "Full pipeline complete!"

clean:
	rm -rf data/processed/ models/ reports/ __pycache__ .pytest_cache
	@echo "Cleaned all generated files."
