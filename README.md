# MovieLens Recommendation System (MLOps)

A production-oriented movie recommendation system using collaborative filtering (ALS) with an optional neural model (Neural Collaborative Filtering). The project includes a FastAPI service for online inference, a Gradio demo UI, reproducible training and evaluation pipelines, and CI/CD with Docker and GitHub Actions.

Important: AWS deployment integration (ECS) is scaffolded but incomplete. Placeholders exist in the workflow and require environment-specific values and infrastructure.

## Overview

This system trains on the MovieLens dataset to provide personalized recommendations and item-to-item similarity. It exposes:

- A FastAPI service with endpoints for user recommendations, similar items, batch prediction, health, and model management
- Offline training and hyperparameter tuning (ALS via implicit; Optuna), tracked with MLflow
- Evaluation utilities that compute Precision@K, Recall@K, and MAP@K on a time-based validation split
- A Gradio demo for manual exploration

## Key Features

- Recommendation models: ALS baseline; optional Neural Collaborative Filtering
- API: Production-style FastAPI app with CORS, trusted hosts, structured error handling, and request metadata
- Model registry: Thread-safe model registry to manage multiple models and set a default serving model
- Training: Configuration-driven ALS training with Optuna sweeps and MLflow tracking
- Neural pipeline: Data preprocessing and training utilities for the neural model
- Evaluation: Precision@K, Recall@K, MAP@K on held-out, time-based validation
- MLOps tooling: Dockerfile, Makefile targets, unit tests, GitHub Actions CI
- Demo: Gradio interface for recommendations

## Project Structure

```
MLOps/
├── Recommender/
│   ├── configs/
│   │   └── model_config.yaml            # Paths, ALS/neural config, CUDA, Optuna, evaluation
│   ├── data/
│   │   ├── raw/                         # MovieLens CSVs (ratings.csv, movies.csv, ...)
│   │   └── processed/                   # Preprocessed train/validation, neural artifacts
│   ├── models/                          # Trained model artifacts (ALS, neural)
│   └── src/
│       ├── api/
│       │   ├── main.py                  # FastAPI app, middleware, router registration
│       │   ├── config.py                # Pydantic settings (env-driven)
│       │   ├── dependencies.py          # DI for model selection, API key checks
│       │   └── routers/                 # /recommendations, /health, /models
│       ├── core/
│       │   ├── interfaces.py            # BaseRecommendationModel, registry interfaces
│       │   ├── schemas.py               # Pydantic request/response models
│       │   └── exceptions.py            # Domain exceptions
│       ├── models/
│       │   ├── als_model.py             # ALS model adapted to standard interface
│       │   ├── neural_model.py          # Neural CF model adapted to interface
│       │   └── registry.py              # Thread-safe model registry
│       ├── explore_data.py              # Raw data exploration helper
│       ├── preprocess_data.py           # Data preprocessing and time-based split
│       ├── train.py                     # ALS training + Optuna + MLflow
│       ├── evaluator.py                 # Evaluation metrics and reporting
│       ├── analyze_validation.py        # Validation set analysis helper
│       ├── predict.py                   # ALS-based local recommendation helper
│       ├── train_neural.py              # Neural model training with MLflow/Optuna
│       └── predict_neural.py            # Neural-based local recommendation helper
├── app.py                               # Gradio demo (local UI)
├── Dockerfile                           # Image for serving API
├── Makefile                             # Common tasks (preprocess/train/evaluate/serve)
├── requirements.txt                     # Runtime/training dependencies
├── requirements-dev.txt                 # Dev tooling (pytest, ruff, etc.)
├── .github/workflows/ci.yml             # CI/CD pipeline (AWS deploy step incomplete)
└── README.md
```

## Setup

Prerequisites:

- Python 3.9+ (Docker image uses 3.9-slim)
- Git
- Optional: CUDA 12.x for GPU-accelerated ALS and neural training

Create environment and install dependencies:

```bash
python -m venv .venv_mlops
. .venv_mlops/Scripts/activate  # Windows PowerShell: .venv_mlops\Scripts\Activate.ps1
pip install -r requirements.txt
```

Data setup:

1) Download MovieLens (ml-latest) from GroupLens and extract CSVs into `Recommender/data/raw/`
2) Required files: `ratings.csv`, `movies.csv` (others optional for this pipeline)

## Usage

Preprocess and split data (time-based, per user):

```bash
python Recommender/src/preprocess_data.py
```

Train ALS with Optuna, track with MLflow:

```bash
python Recommender/src/train.py
```

Evaluate on validation set (Precision@K, Recall@K, MAP@K):

```bash
python Recommender/src/evaluator.py
```

Serve the API locally:

```bash
uvicorn Recommender.src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Gradio demo UI:

```bash
python app.py
```

Makefile shortcuts (optional):

```bash
make preprocess
make train
make evaluate
make serve
```

## Configuration

The file `Recommender/configs/model_config.yaml` controls paths and parameters. Highlights:

- Paths: `processed_data_path`, `model_path`
- MLflow: `mlflow.experiment_name`
- Optuna: `optuna.n_trials`
- CUDA: `cuda.enabled`, `cuda.device`
- ALS: `als.factors`, `als.regularization`, `als.iterations`, `als.random_state`
- Evaluation: `evaluation.k`
- Neural: `neural.epochs`, `neural.embedding_dim`, `neural.layers`, `neural.dropout`, `neural.learning_rate`, `neural.batch_size`

## API

Base URL: `/` (docs at `/docs`, OpenAPI JSON at `/openapi.json`)

- POST `/api/v1/recommendations/user`
  - Body: `{ "user_id": int, "n_recommendations": int, "filter_criteria": { ... } }`
  - Returns ranked recommendations for a user. Cold-start falls back to popular items.

- POST `/api/v1/recommendations/similar`
  - Body: `{ "item_id": int, "n_similar": int }`
  - Returns items similar to the provided item.

- POST `/api/v1/recommendations/batch`
  - Body: `{ "user_ids": [int, ...], "n_recommendations": int }`
  - Returns recommendations for multiple users in one request.

- GET `/api/v1/health`
  - Health/readiness/liveness checks.

- GET `/api/v1/models` and `/api/v1/models/{model_id}`
  - List and inspect registered models; set default via `POST /api/v1/models/{model_id}/set-default`.

Model selection:

- Use a query param `model_id`, header `X-Model-ID`, or the registry default.

## Docker

Build and run the API:

```bash
docker build -t mlops-recsys .
docker run --rm -p 8000:8000 mlops-recsys
```

Mount local data/models if needed:

```bash
docker run --rm -p 8000:8000 \
  -v %cd%\Recommender:/workspace/Recommender \
  mlops-recsys
```

## CI/CD

GitHub Actions workflow:

- Lint (ruff) and test (pytest)
- Build and publish Docker image to GHCR
- AWS ECS deployment step is present but incomplete and contains placeholders. You must supply your task definition, container name, cluster, service, and AWS credentials/secrets and provision the required infrastructure before enabling deployment.

## Notes on Models and Artifacts

- ALS artifacts (model, maps, interaction matrix) are saved under `Recommender/models/`
- The API loads models at startup via the registry; ensure artifacts exist before serving
- Neural model support is optional and depends on `neural_*` artifacts being present


