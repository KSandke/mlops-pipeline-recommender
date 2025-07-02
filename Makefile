.PHONY: help install install-dev test lint format clean train evaluate predict setup-data

# Default target
help:
	@echo "Available targets:"
	@echo "  install      - Install production dependencies"
	@echo "  install-dev  - Install development dependencies"
	@echo "  test         - Run tests with coverage"
	@echo "  lint         - Run linting checks"
	@echo "  format       - Format code with black and isort"
	@echo "  clean        - Clean up generated files"
	@echo "  train        - Train the recommendation model"
	@echo "  evaluate     - Evaluate model performance"
	@echo "  predict      - Generate sample predictions"
	@echo "  setup-data   - Set up data directories"
	@echo "  pipeline     - Run full ML pipeline"

# Installation targets
install:
	pip install -r Recommender/requirements.txt

install-dev:
	pip install -r Recommender/requirements.txt
	pip install -r Recommender/requirements-dev.txt

# Development targets
test:
	pytest tests/ --cov=Recommender/src --cov-report=html --cov-report=term

lint:
	flake8 Recommender/src
	mypy Recommender/src --ignore-missing-imports
	bandit -r Recommender/src

format:
	black Recommender/src
	isort Recommender/src

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf Recommender/models/*.joblib
	rm -rf Recommender/data/processed/*.csv
	rm -rf wandb

# ML Pipeline targets
setup-data:
	mkdir -p Recommender/data/raw
	mkdir -p Recommender/data/processed
	mkdir -p Recommender/models
	@echo "Data directories created. Place MovieLens data in Recommender/data/raw/"

train:
	python Recommender/src/preprocess_data.py
	python Recommender/src/train.py

evaluate:
	python Recommender/src/evaluator.py

predict:
	python Recommender/src/predict.py

# Full pipeline
pipeline: setup-data train evaluate predict
	@echo "Full ML pipeline completed!"

# Docker targets (optional)
docker-build:
	docker build -t movielens-recommender .

docker-run:
	docker run -it --rm -v $(PWD):/workspace movielens-recommender

# Git hooks
install-hooks:
	pre-commit install 