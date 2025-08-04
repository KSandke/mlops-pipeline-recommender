#!/usr/bin/env python3
"""
Preprocess data for neural network training.
This script prepares and saves all the data once, so training can load it instantly.
"""

import pandas as pd
import numpy as np
import os
import joblib
import yaml
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')

def create_feature_encoders(df):
    """Create encoders for categorical features using vectorized operations."""
    encoders = {}
    
    # User ID encoder
    user_encoder = LabelEncoder()
    df['user_id_encoded'] = user_encoder.fit_transform(df['userId'])
    encoders['user'] = user_encoder
    
    # Movie ID encoder
    movie_encoder = LabelEncoder()
    df['movie_id_encoded'] = movie_encoder.fit_transform(df['movieId'])
    encoders['movie'] = movie_encoder
    
    # Genre encoder (vectorized approach)
    # Flatten all genres into a single list efficiently
    all_genres = set()
    
    # Handle both list and string formats efficiently
    genre_series = df['genres'].dropna()
    
    # Process string genres
    string_genres = genre_series[genre_series.apply(lambda x: isinstance(x, str))]
    if not string_genres.empty:
        split_genres = string_genres.str.split('|')
        all_genres.update([genre for genres in split_genres for genre in genres])
    
    # Process list genres
    list_genres = genre_series[genre_series.apply(lambda x: isinstance(x, list))]
    if not list_genres.empty:
        all_genres.update([genre for genres in list_genres for genre in genres])
    
    genre_encoder = LabelEncoder()
    genre_encoder.fit(list(all_genres))
    encoders['genre'] = genre_encoder
    
    return encoders

def create_movie_features_conservative(df, encoders):
    """Create movie feature vectors for UNIQUE movies only - extremely conservative."""
    print("Creating movie features for unique movies only...")
    
    # Get unique movies only - this is the key fix!
    unique_movies = df[['movieId', 'title', 'genres', 'year']].drop_duplicates(subset=['movieId'])
    print(f"Processing {len(unique_movies)} unique movies instead of {len(df)} total rows")
    
    # Initialize feature matrix for unique movies only
    num_unique_movies = len(unique_movies)
    num_genres = len(encoders['genre'].classes_)
    feature_dim = 1 + num_genres  # year + genres
    
    print(f"Creating feature matrix: {num_unique_movies} x {feature_dim} = {num_unique_movies * feature_dim * 4 / 1024**3:.2f} GB")
    
    movie_features = np.zeros((num_unique_movies, feature_dim), dtype=np.float32)
    
    # Create movie ID to index mapping
    movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(unique_movies['movieId'])}
    
    # Year features (vectorized)
    years = unique_movies['year'].fillna(1990).values
    movie_features[:, 0] = (years - 1900) / 100  # Normalize year
    
    # Genre features (conservative approach)
    genre_vectors = np.zeros((num_unique_movies, num_genres), dtype=np.float32)
    
    # Create genre mapping for fast lookup
    genre_to_idx = {genre: idx for idx, genre in enumerate(encoders['genre'].classes_)}
    
    # Process each unique movie's genres
    for idx, (_, movie) in enumerate(unique_movies.iterrows()):
        genres = movie['genres']
        if pd.isna(genres) or genres == '':
            continue
            
        # Handle both string and list formats
        if isinstance(genres, str):
            movie_genres = genres.split('|')
        elif isinstance(genres, list):
            movie_genres = genres
        else:
            continue
        
        # Set genre indicators
        for genre in movie_genres:
            if genre in genre_to_idx:
                genre_vectors[idx, genre_to_idx[genre]] = 1.0
    
    # Combine year and genre features
    movie_features[:, 1:] = genre_vectors
    
    return movie_features, movie_id_to_idx

def preprocess_data(config_path="Recommender/configs/model_config.yaml"):
    """Preprocess and save all data for neural network training."""
    print("🔄 Starting data preprocessing...")
    print(f"📂 Current working directory: {os.getcwd()}")
    
    # Handle config path resolution
    if not os.path.isabs(config_path):
        current_dir = os.getcwd()
        possible_paths = [
            config_path,
            os.path.join(current_dir, config_path),
            os.path.join(current_dir, "..", config_path),
            os.path.join(current_dir, "..", "..", config_path),
        ]
        
        print(f"🔍 Looking for config file in:")
        for path in possible_paths:
            print(f"   - {path} {'✅' if os.path.exists(path) else '❌'}")
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                print(f"✅ Found config at: {config_path}")
                break
        else:
            raise FileNotFoundError(f"Config file not found. Tried: {possible_paths}")
    
    # Load configuration
    print(f"📁 Loading configuration from {config_path}...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        raise
    
    # Handle data path resolution
    processed_data_path = config['processed_data_path']
    print(f"🔍 Looking for processed data at: {processed_data_path}")
    
    if not os.path.isabs(processed_data_path):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        possible_data_paths = [
            processed_data_path,
            os.path.join(config_dir, processed_data_path),
            os.path.join(config_dir, "..", processed_data_path),
        ]
        
        print(f"🔍 Checking possible data paths:")
        for path in possible_data_paths:
            print(f"   - {path} {'✅' if os.path.exists(path) else '❌'}")
        
        for path in possible_data_paths:
            if os.path.exists(path):
                processed_data_path = path
                print(f"✅ Found data at: {processed_data_path}")
                break
        else:
            raise FileNotFoundError(f"Processed data path not found. Tried: {possible_data_paths}")
    
    # Create neural data directory
    neural_data_path = os.path.join(processed_data_path, 'neural')
    os.makedirs(neural_data_path, exist_ok=True)
    
    print(f"📂 Neural data will be saved to: {neural_data_path}")
    
    # Load data with conservative settings
    print("📊 Loading CSV files...")
    train_csv_path = os.path.join(processed_data_path, 'train.csv')
    val_csv_path = os.path.join(processed_data_path, 'validation.csv')
    
    print(f"🔍 Looking for train.csv at: {train_csv_path}")
    print(f"🔍 Looking for validation.csv at: {val_csv_path}")
    
    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"train.csv not found at: {train_csv_path}")
    if not os.path.exists(val_csv_path):
        raise FileNotFoundError(f"validation.csv not found at: {val_csv_path}")
    
    # Load data without specifying dtypes first to avoid conflicts
    print("✅ CSV files found, loading data...")
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(val_csv_path)
    
    print(f"✅ Loaded {len(train_df)} training samples, {len(val_df)} validation samples")
    print(f"   - Training data shape: {train_df.shape}")
    print(f"   - Training data columns: {list(train_df.columns)}")
    print(f"   - Training data dtypes: {train_df.dtypes.to_dict()}")
    
    # Create encoders from combined data to handle all user/movie IDs
    print("🔧 Creating feature encoders from combined data...")
    
    # Combine user and movie IDs to ensure all are seen during encoding
    all_user_ids = pd.concat([train_df['userId'], val_df['userId']]).unique()
    all_movie_ids = pd.concat([train_df['movieId'], val_df['movieId']]).unique()
    
    # Create encoders that know about all possible IDs
    user_encoder = LabelEncoder()
    user_encoder.fit(all_user_ids)
    
    movie_encoder = LabelEncoder()
    movie_encoder.fit(all_movie_ids)
    
    # Create genre encoder from training data only (genres should be consistent)
    all_genres = set()
    genre_series = train_df['genres'].dropna()
    
    # Process string genres
    string_genres = genre_series[genre_series.apply(lambda x: isinstance(x, str))]
    if not string_genres.empty:
        split_genres = string_genres.str.split('|')
        all_genres.update([genre for genres in split_genres for genre in genres])
    
    # Process list genres
    list_genres = genre_series[genre_series.apply(lambda x: isinstance(x, list))]
    if not list_genres.empty:
        all_genres.update([genre for genres in list_genres for genre in genres])
    
    genre_encoder = LabelEncoder()
    genre_encoder.fit(list(all_genres))
    
    # Store encoders
    encoders = {
        'user': user_encoder,
        'movie': movie_encoder,
        'genre': genre_encoder
    }
    
    print("✅ Encoders created successfully")
    
    # Apply encoders to both datasets
    print("🔧 Applying encoders to training data...")
    train_df['user_id_encoded'] = encoders['user'].transform(train_df['userId'])
    train_df['movie_id_encoded'] = encoders['movie'].transform(train_df['movieId'])
    
    print("🔧 Applying encoders to validation data...")
    val_df['user_id_encoded'] = encoders['user'].transform(val_df['userId'])
    val_df['movie_id_encoded'] = encoders['movie'].transform(val_df['movieId'])
    print("✅ Encoders applied to both datasets")
    
    # Create movie features for unique movies only
    print("🎬 Creating movie features for unique movies...")
    movie_features, movie_id_to_idx = create_movie_features_conservative(train_df, encoders)
    print("✅ Movie features created")
    
    # Prepare training data - use movie indices instead of full features
    print("⚙️ Preparing training data...")
    train_data = {
        'user_ids': train_df['user_id_encoded'].values,
        'movie_ids': train_df['movie_id_encoded'].values,
        'ratings': train_df['rating'].values,
        'movie_features': movie_features,  # Features for unique movies only
        'movie_id_to_idx': movie_id_to_idx  # Mapping to get features
    }
    
    # Prepare validation data
    print("⚙️ Preparing validation data...")
    val_data = {
        'user_ids': val_df['user_id_encoded'].values,
        'movie_ids': val_df['movie_id_encoded'].values,
        'ratings': val_df['rating'].values,
        'movie_features': movie_features,  # Same features, shared
        'movie_id_to_idx': movie_id_to_idx  # Same mapping, shared
    }
    
    # Clear memory
    del train_df, val_df
    gc.collect()
    
    # Save everything
    print("💾 Saving preprocessed data...")
    
    # Save encoders
    joblib.dump(encoders, os.path.join(neural_data_path, 'encoders.joblib'))
    
    # Save training data
    joblib.dump(train_data, os.path.join(neural_data_path, 'train_data.joblib'))
    
    # Save validation data
    joblib.dump(val_data, os.path.join(neural_data_path, 'val_data.joblib'))
    
    # Save metadata
    metadata = {
        'num_users': len(encoders['user'].classes_),
        'num_movies': len(encoders['movie'].classes_),
        'num_genres': len(encoders['genre'].classes_),
        'movie_feature_dim': movie_features.shape[1],
        'train_samples': len(train_data['user_ids']),
        'val_samples': len(val_data['user_ids']),
        'unique_movies': len(movie_features)
    }
    joblib.dump(metadata, os.path.join(neural_data_path, 'metadata.joblib'))
    
    print("✅ Data preprocessing completed!")
    print(f"📊 Dataset info:")
    print(f"   - Users: {metadata['num_users']}")
    print(f"   - Movies: {metadata['num_movies']}")
    print(f"   - Unique movies with features: {metadata['unique_movies']}")
    print(f"   - Genres: {metadata['num_genres']}")
    print(f"   - Movie features: {metadata['movie_feature_dim']} dimensions")
    print(f"   - Training samples: {metadata['train_samples']}")
    print(f"   - Validation samples: {metadata['val_samples']}")
    print(f"📁 All data saved to: {neural_data_path}")

if __name__ == '__main__':
    preprocess_data() 