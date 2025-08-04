import pandas as pd
import scipy.sparse as sparse
import implicit
import os
import joblib
import yaml
import mlflow
import optuna
import numpy as np
from mlflow.models import infer_signature

def load_data(processed_data_path):
    """Loads preprocessed training data."""
    print("Loading processed data...")
    train_df = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
    
    train_df['userId'] = train_df['userId'].astype("category")
    train_df['movieId'] = train_df['movieId'].astype("category")

    user_map = train_df['userId'].cat.categories
    movie_map = train_df['movieId'].cat.categories

    interaction_matrix = sparse.csr_matrix(
        (train_df['rating'].astype(float),
         (train_df['userId'].cat.codes,
          train_df['movieId'].cat.codes))
    )
    
    return train_df, interaction_matrix, user_map, movie_map

def objective(trial, config, interaction_matrix):
    """
    The objective function for Optuna to minimize.
    Trains a model with a given set of hyperparameters and logs it with MLflow.
    """
    # Define hyperparameter search space
    factors = trial.suggest_int('factors', 64, 512)
    regularization = trial.suggest_float('regularization', 0.001, 0.1, log=True)
    alpha = trial.suggest_float('alpha', 1.0, 40.0, log=True) # Recommended by implicit docs
    iterations = trial.suggest_int('iterations', 15, 50)

    # Start an MLflow run
    with mlflow.start_run():
        mlflow.log_params(trial.params)
        
        # Log hardware config
        use_gpu = config.get('cuda', {}).get('enabled', False)
        mlflow.log_param("use_gpu", use_gpu)

        # Initialize and train the ALS model
        model = implicit.als.AlternatingLeastSquares(
            factors=factors,
            regularization=regularization,
            alpha=alpha,
            iterations=iterations,
            calculate_training_loss=True,
            random_state=config['als']['random_state'],
            use_gpu=use_gpu
        )
        
        print(f"Training with params: {trial.params}")
        
        # Use a callback to capture training loss, as the `training_loss_` attribute
        # has been removed in recent versions of `implicit`.
        losses = []
        def fit_callback(iteration, elapsed, loss):
            losses.append(loss)

        model.fit(interaction_matrix, callback=fit_callback)

        # For this example, we'll use the final training loss as the metric to minimize.
        # In a real scenario, you would use a proper validation metric (e.g., precision@k, NDCG).
        final_loss = losses[-1] if losses else float('inf')
        mlflow.log_metric("final_training_loss", final_loss)
        
        # Create a dummy input for signature inference
        dummy_input = {
            "user_id": np.array([0]),
            "n_recs": np.array([10])
        }
        
        # Log the model with a signature
        signature = infer_signature(dummy_input)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="als_model",
            signature=signature,
            registered_model_name="als-recommender" # Optionally register the model
        )

    return final_loss


def run_hyperparameter_tuning(config_path="Recommender/configs/model_config.yaml"):
    """
    Runs hyperparameter tuning using Optuna and logs experiments with MLflow.
    """
    print("Starting hyperparameter tuning...")

    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load data
    _, interaction_matrix, user_map, movie_map = load_data(config['processed_data_path'])

    # Set up MLflow experiment
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Set up and run Optuna study
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda trial: objective(trial, config, interaction_matrix), 
        n_trials=config['optuna']['n_trials']
    )

    print("Hyperparameter tuning finished.")
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best value: {study.best_value}")

    # Log the best trial to MLflow in a separate, summary run
    with mlflow.start_run(run_name="best_hyperparameters"):
        mlflow.log_params(study.best_trial.params)
        mlflow.log_metric("best_final_loss", study.best_value)

        # Train the best model and save it
        print("Training and saving the best model...")
        best_params = study.best_trial.params
        model = implicit.als.AlternatingLeastSquares(
            factors=best_params['factors'],
            regularization=best_params['regularization'],
            alpha=best_params['alpha'],
            iterations=best_params['iterations'],
            calculate_training_loss=True,
            random_state=config['als']['random_state'],
            use_gpu=config.get('cuda', {}).get('enabled', False)
        )
        model.fit(interaction_matrix)

        # Save the final model artifacts
        model_path = config['model_path']
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            
        joblib.dump(model, os.path.join(model_path, 'als_model.joblib'))
        joblib.dump(user_map, os.path.join(model_path, 'user_map.joblib'))
        joblib.dump(movie_map, os.path.join(model_path, 'movie_map.joblib'))
        joblib.dump(interaction_matrix, os.path.join(model_path, 'interaction_matrix.joblib'))
        print(f"Best model artifacts saved to {model_path}")

if __name__ == '__main__':
    run_hyperparameter_tuning() 