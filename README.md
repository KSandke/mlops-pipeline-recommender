# MovieLens Recommendation System - MLOps Project

A production-ready movie recommendation system built with collaborative filtering, featuring comprehensive MLOps practices including experiment tracking, model evaluation, and reproducible pipelines.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![MLOps](https://img.shields.io/badge/MLOps-Enabled-green.svg)
![Weights & Biases](https://img.shields.io/badge/Weights%20&%20Biases-Tracking-orange.svg)

## 🎯 Project Overview

This project implements a collaborative filtering recommendation system using the MovieLens dataset with modern MLOps practices. The system can recommend movies to users based on their historical ratings and find similar movies using matrix factorization techniques.

### Key Features

- **Collaborative Filtering**: ALS (Alternating Least Squares) matrix factorization
- **Experiment Tracking**: Weights & Biases integration
- **Model Evaluation**: Comprehensive metrics (Precision@K, Recall@K, MAP@K)
- **GPU Acceleration**: CUDA support for faster training
- **Reproducible Pipelines**: Configuration-driven training and evaluation
- **Production Ready**: Modular code structure with proper error handling

## 📊 Performance Metrics (Baseline)

| Metric | Value |
|--------|-------|
| Precision@10 | 0.0144 (1.44%) |
| Recall@10 | 0.1436 (14.36%) |
| MAP@10 | 0.0499 |

*Evaluated on 564 users with ratings ≥4.0 from time-based validation split*

## 🏗️ Project Structure

```
MLOps/
├── Recommender/
│   ├── configs/
│   │   └── model_config.yaml          # Model hyperparameters and paths
│   ├── data/
│   │   ├── raw/                       # Original MovieLens data
│   │   ├── processed/                 # Preprocessed train/validation sets
│   │   └── splits/                    # Data splitting artifacts
│   ├── models/                        # Trained model artifacts
│   ├── notebooks/                     # Jupyter notebooks for exploration
│   ├── src/
│   │   ├── data_loader.py            # Data loading utilities
│   │   ├── explore_data.py           # Data exploration script
│   │   ├── preprocess_data.py        # Data preprocessing pipeline
│   │   ├── train.py                  # Model training script
│   │   ├── predict.py                # Inference and recommendation script
│   │   ├── evaluator.py              # Model evaluation metrics
│   │   └── analyze_validation.py     # Validation set analysis
│   └── requirements.txt              # Python dependencies
├── .gitignore                        # Git ignore rules for ML projects
└── README.md                         # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA Toolkit 12.x (optional, for GPU acceleration)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd MLOps
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv Recommender/.venv
   
   # Windows
   .\Recommender\.venv\Scripts\Activate.ps1
   
   # Linux/Mac
   source Recommender/.venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r Recommender/requirements.txt
   ```

4. **Set up Weights & Biases** (optional)
   ```bash
   wandb login
   ```

### Data Setup

1. Download the MovieLens Latest Dataset from [GroupLens](https://grouplens.org/datasets/movielens/)
2. Extract CSV files to `Recommender/data/raw/`
3. Required files:
   - `ratings.csv`
   - `movies.csv`
   - `tags.csv`
   - `links.csv`
   - `genome-scores.csv`
   - `genome-tags.csv`

## 🔄 Usage

### 1. Data Exploration
```bash
python Recommender/src/explore_data.py
```

### 2. Data Preprocessing
```bash
python Recommender/src/preprocess_data.py
```

### 3. Model Training
```bash
python Recommender/src/train.py
```

### 4. Model Evaluation
```bash
python Recommender/src/evaluator.py
```

### 5. Generate Recommendations
```bash
python Recommender/src/predict.py
```

## ⚙️ Configuration

Edit `Recommender/configs/model_config.yaml` to customize:

- **Model hyperparameters**: factors, regularization, iterations
- **CUDA settings**: GPU acceleration on/off
- **Weights & Biases**: project name, entity, tags
- **File paths**: data and model directories

```yaml
# ALS Model Hyperparameters
als:
  factors: 50
  regularization: 0.01
  iterations: 20
  random_state: 42

# CUDA Configuration
cuda:
  enabled: true
  device: 0
```

## 📈 Experiment Tracking

The project uses Weights & Biases for experiment tracking:

- **Hyperparameters**: All model configuration logged
- **Training Metrics**: Loss progression, dataset statistics
- **Model Artifacts**: Trained models automatically saved
- **System Metrics**: GPU usage, training time

View your experiments at: https://wandb.ai/your-username/movielens-recommender

## 🧪 Model Evaluation

The evaluation system provides comprehensive metrics:

- **Precision@K**: Fraction of recommended items that are relevant
- **Recall@K**: Fraction of relevant items that are recommended  
- **MAP@K**: Mean Average Precision (accounts for ranking quality)

Evaluation uses time-based validation split where each user's most recent rating is held out for testing.

## 🔧 API Usage

### Generate User Recommendations
```python
from Recommender.src.predict import MovieRecommender

recommender = MovieRecommender()
recommendations = recommender.get_user_recommendations(
    user_id=123, 
    n_recommendations=10
)
print(recommendations)
```

### Find Similar Movies
```python
similar_movies = recommender.get_similar_movies(
    movie_id=1, 
    n_similar=5
)
print(similar_movies)
```

## 🚀 GPU Acceleration

For faster training with CUDA:

1. **Install CUDA-enabled dependencies**
   ```bash
   pip install cupy-cuda12x  # For CUDA 12.x
   ```

2. **Enable in configuration**
   ```yaml
   cuda:
     enabled: true
     device: 0
   ```

Expected speedup: 5-10x faster training on compatible GPUs.

## 📊 Dataset Information

- **Dataset**: MovieLens Latest (ml-latest)
- **Ratings**: 33M+ ratings from 330K+ users
- **Movies**: 86K+ movies with metadata
- **Time Range**: 1995-2023
- **Rating Scale**: 0.5-5.0 stars

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [GroupLens Research](https://grouplens.org/) for the MovieLens dataset
- [Implicit Library](https://github.com/benfred/implicit) for collaborative filtering algorithms
- [Weights & Biases](https://wandb.ai/) for experiment tracking

## 📚 References

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems, 5(4), 1-19.
- Hu, Y., Koren, Y., & Volinsky, C. (2008). Collaborative Filtering for Implicit Feedback Datasets. IEEE International Conference on Data Mining.

---

**Built with ❤️ for the MLOps community** 