import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def preprocess_data(raw_data_path="Recommender/data/raw", processed_data_path="Recommender/data/processed"):
    """
    Loads raw data, preprocesses it, and splits it into training and validation sets.
    """
    print("Starting data preprocessing...")

    # Load data
    print("Loading raw data...")
    ratings_df = pd.read_csv(os.path.join(raw_data_path, 'ratings.csv'))
    movies_df = pd.read_csv(os.path.join(raw_data_path, 'movies.csv'))

    # Merge ratings and movies data
    df = pd.merge(ratings_df, movies_df, on='movieId')

    # Feature Engineering
    print("Performing feature engineering...")
    # Extract year from title
    df['year'] = df['title'].str.extract(r'\((\d{4})\)', expand=False)
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    # Remove year from title
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)$', '', regex=True)

    # Split genres string into a list
    df['genres'] = df['genres'].str.split('|')

    print("Data after feature engineering:")
    print(df.head())

    # Time-based split
    print("Performing user-centric, time-based split...")
    df = df.sort_values(by=['userId', 'timestamp'])
    
    train_data = []
    val_data = []

    for user_id, group in df.groupby('userId'):
        # Use all but the last rating for training
        train_data.append(group.iloc[:-1])
        # Use the last rating for validation
        val_data.append(group.iloc[-1:])

    train_df = pd.concat(train_data)
    val_df = pd.concat(val_data)

    print(f"\nTraining set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")

    # Save processed data
    print(f"Saving processed data to {processed_data_path}...")
    if not os.path.exists(processed_data_path):
        os.makedirs(processed_data_path)
        
    train_df.to_csv(os.path.join(processed_data_path, 'train.csv'), index=False)
    val_df.to_csv(os.path.join(processed_data_path, 'validation.csv'), index=False)

    print("Data preprocessing finished successfully.")

if __name__ == '__main__':
    preprocess_data() 