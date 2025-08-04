import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import joblib
import yaml
import mlflow
import optuna
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings
from torch.cuda.amp import autocast, GradScaler
warnings.filterwarnings('ignore')

class MovieDataset(Dataset):
    """Custom dataset for movie recommendation data."""
    
    def __init__(self, user_ids, movie_ids, ratings, user_features=None, movie_features=None, movie_id_to_idx=None):
        self.user_ids = torch.LongTensor(user_ids)
        self.movie_ids = torch.LongTensor(movie_ids)
        self.ratings = torch.FloatTensor(ratings)
        self.user_features = user_features
        self.movie_features = movie_features
        self.movie_id_to_idx = movie_id_to_idx
        
    def __len__(self):
        return len(self.ratings)
    
    def __getitem__(self, idx):
        item = {
            'user_id': self.user_ids[idx],
            'movie_id': self.movie_ids[idx],
            'rating': self.ratings[idx]
        }
        
        if self.user_features is not None:
            item['user_features'] = torch.FloatTensor(self.user_features[idx])
        if self.movie_features is not None and self.movie_id_to_idx is not None:
            # Use movie_id to get the correct feature index
            movie_id = self.movie_ids[idx].item()
            feature_idx = self.movie_id_to_idx.get(movie_id, 0)  # Default to 0 if not found
            item['movie_features'] = torch.FloatTensor(self.movie_features[feature_idx])
            
        return item

class NeuralCollaborativeFiltering(nn.Module):
    """Neural Collaborative Filtering model."""
    
    def __init__(self, num_users, num_movies, embedding_dim=64, layers=[128, 64, 32], dropout=0.1):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        # Embedding layers
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Feature dimensions (if using additional features)
        self.user_feature_dim = 0  # Will be set if user features are provided
        self.movie_feature_dim = 0  # Will be set if movie features are provided
        
        # MLP layers
        input_dim = embedding_dim * 2  # user + movie embeddings
        self.mlp_layers = nn.ModuleList()
        
        for layer_size in layers:
            self.mlp_layers.append(nn.Linear(input_dim, layer_size))
            self.mlp_layers.append(nn.ReLU())
            self.mlp_layers.append(nn.Dropout(dropout))
            input_dim = layer_size
        
        # Output layer
        self.output_layer = nn.Linear(input_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
    
    def forward(self, user_ids, movie_ids, user_features=None, movie_features=None):
        # Get embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Concatenate embeddings
        concat_features = torch.cat([user_emb, movie_emb], dim=1)
        
        # Pass through MLP
        x = concat_features
        for layer in self.mlp_layers:
            x = layer(x)
        
        # Output layer
        output = self.output_layer(x)
        return output.squeeze()

def create_feature_encoders(df):
    """Create encoders for categorical features."""
    encoders = {}
    
    # User ID encoder
    user_encoder = LabelEncoder()
    df['user_id_encoded'] = user_encoder.fit_transform(df['userId'])
    encoders['user'] = user_encoder
    
    # Movie ID encoder
    movie_encoder = LabelEncoder()
    df['movie_id_encoded'] = movie_encoder.fit_transform(df['movieId'])
    encoders['movie'] = movie_encoder
    
    # Genre encoder (multi-hot encoding)
    all_genres = set()
    for genres in df['genres'].dropna():
        if isinstance(genres, list):
            all_genres.update(genres)
    
    genre_encoder = LabelEncoder()
    genre_encoder.fit(list(all_genres))
    encoders['genre'] = genre_encoder
    
    return encoders

def create_movie_features(df, encoders):
    """Create movie feature vectors."""
    movie_features = []
    
    for _, row in df.iterrows():
        features = []
        
        # Year feature (normalized)
        year = row.get('year', 1990)
        if pd.isna(year):
            year = 1990
        features.append((year - 1900) / 100)  # Normalize year
        
        # Genre features (multi-hot encoding)
        genres = row.get('genres', [])
        if isinstance(genres, str):
            genres = genres.split('|')
        elif not isinstance(genres, list):
            genres = []
        
        genre_vector = np.zeros(len(encoders['genre'].classes_))
        for genre in genres:
            if genre in encoders['genre'].classes_:
                idx = encoders['genre'].transform([genre])[0]
                genre_vector[idx] = 1
        
        features.extend(genre_vector)
        movie_features.append(features)
    
    return np.array(movie_features)

def load_neural_data(processed_data_path, config):
    """Load preprocessed data for neural network training."""
    print("Loading preprocessed data for neural network training...")
    
    # Check if preprocessed data exists
    neural_data_path = os.path.join(processed_data_path, 'neural')
    if not os.path.exists(neural_data_path):
        print("ERROR: Preprocessed data not found. Please run preprocess_neural_data.py first.")
        print("   Run: python preprocess_neural_data.py")
        raise FileNotFoundError(f"Neural data directory not found: {neural_data_path}")
    
    # Load preprocessed data
    print("Loading preprocessed data...")
    encoders = joblib.load(os.path.join(neural_data_path, 'encoders.joblib'))
    train_data = joblib.load(os.path.join(neural_data_path, 'train_data.joblib'))
    val_data = joblib.load(os.path.join(neural_data_path, 'val_data.joblib'))
    metadata = joblib.load(os.path.join(neural_data_path, 'metadata.joblib'))
    
    print(f"SUCCESS: Loaded preprocessed data:")
    print(f"   - Users: {metadata['num_users']}")
    print(f"   - Movies: {metadata['num_movies']}")
    print(f"   - Training samples: {metadata['train_samples']}")
    print(f"   - Validation samples: {metadata['val_samples']}")
    
    # Create datasets
    train_dataset = MovieDataset(
        train_data['user_ids'], 
        train_data['movie_ids'], 
        train_data['ratings'], 
        movie_features=train_data['movie_features'],
        movie_id_to_idx=train_data['movie_id_to_idx']
    )
    val_dataset = MovieDataset(
        val_data['user_ids'], 
        val_data['movie_ids'], 
        val_data['ratings'],
        movie_features=val_data['movie_features'],
        movie_id_to_idx=val_data['movie_id_to_idx']
    )
    
    return train_dataset, val_dataset, encoders, metadata['num_users'], metadata['num_movies']

def train_epoch(model, dataloader, criterion, optimizer, device, epoch=None, scaler=None):
    """Train for one epoch with mixed precision."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}" if epoch is not None else "Training", 
                leave=False, unit="batch")
    
    for batch in pbar:
        user_ids = batch['user_id'].to(device)
        movie_ids = batch['movie_id'].to(device)
        ratings = batch['rating'].to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast():
            predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def validate_epoch(model, dataloader, criterion, device, epoch=None):
    """Validate for one epoch with mixed precision."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}" if epoch is not None else "Validation", 
                leave=False, unit="batch")
    
    with torch.no_grad():
        for batch in pbar:
            user_ids = batch['user_id'].to(device)
            movie_ids = batch['movie_id'].to(device)
            ratings = batch['rating'].to(device)
            
            # Mixed precision validation
            with autocast():
                predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches

def objective(trial, config, train_dataset, val_dataset, num_users, num_movies):
    """Optuna objective function for hyperparameter tuning."""
    
    # Hyperparameter search space
    embedding_dim = trial.suggest_int('embedding_dim', 32, 128)
    layers = trial.suggest_categorical('layers', ['[64, 32]', '[128, 64, 32]', '[256, 128, 64]'])
    layers = eval(layers)  # Convert string to list
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    # Use config batch size instead of Optuna suggestions
    batch_size = config['neural']['batch_size']
    
    # Device
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA config enabled: {config.get('cuda', {}).get('enabled', False)}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('cuda', {}).get('enabled', False) else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,  # Increased from 0
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=4,  # Increased from 0
        pin_memory=True,  # Faster GPU transfer
        persistent_workers=True  # Keep workers alive
    )
    
    # Initialize model
    model = NeuralCollaborativeFiltering(
        num_users=num_users,
        num_movies=num_movies,
        embedding_dim=embedding_dim,
        layers=layers,
        dropout=dropout
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    with mlflow.start_run():
        mlflow.log_params(trial.params)
        mlflow.log_param("device", str(device))
        
        # Initialize mixed precision scaler
        scaler = GradScaler()
        
        # Main training loop with progress bar
        epoch_pbar = tqdm(range(config['neural']['epochs']), desc="Training Progress", unit="epoch")
        
        for epoch in epoch_pbar:
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch + 1, scaler)
            val_loss = validate_epoch(model, val_loader, criterion, device, epoch + 1)
            
            # Update epoch progress bar
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'best_val': f'{best_val_loss:.4f}'
            })
            
            mlflow.log_metric(f"train_loss_epoch_{epoch}", train_loss)
            mlflow.log_metric(f"val_loss_epoch_{epoch}", val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                epoch_pbar.set_postfix({'status': 'Early stopping'})
                break
        
        # Log best validation loss
        mlflow.log_metric("best_val_loss", best_val_loss)
        
        # Save model
        model_path = os.path.join(config['model_path'], 'ncf_model.pth')
        torch.save(model.state_dict(), model_path)
        mlflow.log_artifact(model_path)
    
    return best_val_loss

def run_neural_training(config_path="Recommender/configs/model_config.yaml"):
    """Run neural network training with hyperparameter tuning."""
    print("Starting neural network training...")
    
    # Set MLflow tracking URI
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # Handle config path resolution
    if not os.path.isabs(config_path):
        # If relative path, try to find it relative to current directory or project root
        current_dir = os.getcwd()
        possible_paths = [
            config_path,  # Try as-is
            os.path.join(current_dir, config_path),  # Try relative to current dir
            os.path.join(current_dir, "..", config_path),  # Try one level up
            os.path.join(current_dir, "..", "..", config_path),  # Try two levels up
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            raise FileNotFoundError(f"Config file not found. Tried: {possible_paths}")
    
    # Load configuration
    print(f"Loading configuration from {config_path}...")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle data path resolution
    processed_data_path = config['processed_data_path']
    if not os.path.isabs(processed_data_path):
        # Try to find data relative to config file location
        config_dir = os.path.dirname(os.path.abspath(config_path))
        possible_data_paths = [
            processed_data_path,  # Try as-is
            os.path.join(config_dir, processed_data_path),  # Try relative to config
            os.path.join(config_dir, "..", processed_data_path),  # Try one level up from config
        ]
        
        for path in possible_data_paths:
            if os.path.exists(path):
                processed_data_path = path
                break
        else:
            raise FileNotFoundError(f"Processed data path not found. Tried: {possible_data_paths}")
    
    # Load data
    train_dataset, val_dataset, encoders, num_users, num_movies = load_neural_data(
        processed_data_path, config
    )
    
    # Set up MLflow experiment
    mlflow.set_experiment(config['mlflow']['experiment_name'] + "_neural")
    
    # Set up and run Optuna study with progress bar
    study = optuna.create_study(direction='minimize')
    
    print(f"\nStarting hyperparameter optimization with {config['optuna']['n_trials']} trials...")
    
    # Create progress bar for trials
    trial_pbar = tqdm(range(config['optuna']['n_trials']), desc="Optuna Trials", unit="trial")
    
    def objective_with_progress(trial):
        result = objective(trial, config, train_dataset, val_dataset, num_users, num_movies)
        trial_pbar.update(1)
        # Update trial progress bar (with error handling)
        try:
            best_loss = study.best_value if study.best_value != float('inf') else 'N/A'
            trial_pbar.set_postfix({'best_loss': f'{best_loss:.4f}' if best_loss != 'N/A' else 'N/A'})
        except ValueError:
            trial_pbar.set_postfix({'best_loss': 'N/A'})
        return result
    
    study.optimize(objective_with_progress, n_trials=config['optuna']['n_trials'])
    trial_pbar.close()
    
    print("Neural network training finished.")
    print(f"Best trial: {study.best_trial.params}")
    print(f"Best validation loss: {study.best_value}")
    
    # Handle model path resolution
    model_path = config['model_path']
    if not os.path.isabs(model_path):
        # Try to find model path relative to config file location
        config_dir = os.path.dirname(os.path.abspath(config_path))
        possible_model_paths = [
            model_path,  # Try as-is
            os.path.join(config_dir, model_path),  # Try relative to config
            os.path.join(config_dir, "..", model_path),  # Try one level up from config
        ]
        
        for path in possible_model_paths:
            if os.path.exists(path) or os.path.exists(os.path.dirname(path)):
                model_path = path
                break
        else:
            # If no existing path found, use relative to config
            model_path = os.path.join(config_dir, "..", model_path)
    
    # Save encoders to model path (for inference)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    joblib.dump(encoders, os.path.join(model_path, 'neural_encoders.joblib'))
    print(f"Encoders saved to {model_path}")
    
    # Also save to neural data directory for consistency
    neural_data_path = os.path.join(processed_data_path, 'neural')
    joblib.dump(encoders, os.path.join(neural_data_path, 'encoders.joblib'))
    print(f"Encoders also saved to {neural_data_path}")

if __name__ == '__main__':
    run_neural_training() 