import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
import os
import joblib
import yaml
import wandb
import shutil

def train_model(config_path="Recommender/configs/model_config.yaml"):
    """
    Trains a collaborative filtering model using the implicit library
    and saves the model artifacts based on a config file.
    """
    print("Starting model training...")

    # Load configuration from YAML file
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    processed_data_path = config['processed_data_path']
    model_path = config['model_path']
    als_params = config['als']
    wandb_config = config['wandb']
    cuda_config = config.get('cuda', {'enabled': False, 'device': 0})

    # Check CUDA availability and configuration
    use_gpu = cuda_config['enabled']
    try:
        import cupy
        gpu_available = True
        print(f"CUDA is available. GPU usage: {'Enabled' if use_gpu else 'Disabled (by config)'}")
    except ImportError:
        gpu_available = False
        use_gpu = False
        print("CUDA is not available. Using CPU.")

    # Initialize wandb if enabled
    if wandb_config['enabled']:
        wandb.init(
            project=wandb_config['project'],
            entity=wandb_config['entity'],
            tags=wandb_config['tags'],
            config={
                'als_factors': als_params['factors'],
                'als_regularization': als_params['regularization'],
                'als_iterations': als_params['iterations'],
                'als_random_state': als_params['random_state'],
                'cuda_enabled': use_gpu,
                'gpu_available': gpu_available
            }
        )

    # Load the processed training data
    print("Loading processed data...")
    train_df = pd.read_csv(os.path.join(processed_data_path, 'train.csv'))
    
    # Log dataset statistics
    if wandb_config['enabled']:
        wandb.log({
            'dataset_size': len(train_df),
            'unique_users': train_df['userId'].nunique(),
            'unique_movies': train_df['movieId'].nunique(),
            'avg_rating': train_df['rating'].mean(),
            'rating_std': train_df['rating'].std()
        })

    # Create user and item mappings to integer indices
    # This is a crucial step because the sparse matrix needs integer indices.
    train_df['userId'] = train_df['userId'].astype("category")
    train_df['movieId'] = train_df['movieId'].astype("category")

    user_map = train_df['userId'].cat.categories
    movie_map = train_df['movieId'].cat.categories
    
    # Create the user-item sparse matrix
    print("Creating user-item interaction matrix...")
    interaction_matrix = sparse.csr_matrix(
        (train_df['rating'].astype(float),
         (train_df['userId'].cat.codes,
          train_df['movieId'].cat.codes))
    )

    # Log matrix sparsity
    if wandb_config['enabled']:
        sparsity = 1.0 - (interaction_matrix.nnz / (interaction_matrix.shape[0] * interaction_matrix.shape[1]))
        wandb.log({
            'matrix_shape': interaction_matrix.shape,
            'matrix_sparsity': sparsity,
            'matrix_nnz': interaction_matrix.nnz
        })

    # Initialize the Alternating Least Squares (ALS) model from config
    print(f"Initializing and training the ALS model from config (GPU: {use_gpu})...")
    model = implicit.als.AlternatingLeastSquares(
        factors=als_params['factors'], 
        regularization=als_params['regularization'], 
        iterations=als_params['iterations'], 
        calculate_training_loss=True,
        random_state=als_params['random_state'],
        use_gpu=use_gpu
    )

    # Train the model
    print("Training model...")
    model.fit(interaction_matrix)

    # Log training loss if available
    if wandb_config['enabled'] and hasattr(model, 'training_loss_'):
        for i, loss in enumerate(model.training_loss_):
            wandb.log({'training_loss': loss, 'iteration': i})

    # Save the model and mappings
    print(f"Saving model artifacts to {model_path}...")
    if not os.path.exists(model_path):
        os.makedirs(model_path)
        
    # Using joblib to save the model and mappings
    model_file = os.path.join(model_path, 'als_model.joblib')
    joblib.dump(model, model_file)
    joblib.dump(user_map, os.path.join(model_path, 'user_map.joblib'))
    joblib.dump(movie_map, os.path.join(model_path, 'movie_map.joblib'))
    joblib.dump(interaction_matrix, os.path.join(model_path, 'interaction_matrix.joblib'))

    # Log model artifacts to wandb (Windows-friendly way)
    if wandb_config['enabled']:
        try:
            # Copy file to wandb directory instead of using symlinks
            wandb_model_file = os.path.join(wandb.run.dir, 'als_model.joblib')
            shutil.copy2(model_file, wandb_model_file)
            wandb.log({'model_saved': True})
        except Exception as e:
            print(f"Warning: Could not save model to wandb: {e}")
            wandb.log({'model_saved': False, 'save_error': str(e)})

    print("Model training finished successfully.")
    
    # Finish wandb run
    if wandb_config['enabled']:
        wandb.finish()

if __name__ == '__main__':
    train_model() 