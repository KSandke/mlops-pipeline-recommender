import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from train_neural import NeuralCollaborativeFiltering, create_movie_features

class NeuralMovieRecommender:
    def __init__(self, model_path="Recommender/models"):
        """
        Initialize the neural recommender by loading the trained model and encoders.
        """
        print("Loading neural model and encoders...")
        
        # Load encoders
        self.encoders = joblib.load(os.path.join(model_path, 'neural_encoders.joblib'))
        
        # Load model architecture and weights
        model_state = torch.load(os.path.join(model_path, 'ncf_model.pth'), map_location='cpu')
        
        # Get model dimensions from encoders
        num_users = len(self.encoders['user'].classes_)
        num_movies = len(self.encoders['movie'].classes_)
        
        # Initialize model with same architecture
        self.model = NeuralCollaborativeFiltering(
            num_users=num_users,
            num_movies=num_movies,
            embedding_dim=64,  # Default, will be overridden by loaded weights
            layers=[128, 64, 32],
            dropout=0.1
        )
        
        # Load trained weights
        self.model.load_state_dict(model_state)
        self.model.eval()
        
        # Load movie metadata
        self.movies_df = pd.read_csv("Recommender/data/raw/movies.csv")
        
        print("Neural model loaded successfully!")
        print(f"- Users in training: {num_users}")
        print(f"- Movies in training: {num_movies}")
    
    def _encode_user_id(self, user_id):
        """Encode user ID to internal representation."""
        if user_id not in self.encoders['user'].classes_:
            return None
        return self.encoders['user'].transform([user_id])[0]
    
    def _encode_movie_id(self, movie_id):
        """Encode movie ID to internal representation."""
        if movie_id not in self.encoders['movie'].classes_:
            return None
        return self.encoders['movie'].transform([movie_id])[0]
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a user-movie pair.
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            Predicted rating
        """
        user_encoded = self._encode_user_id(user_id)
        movie_encoded = self._encode_movie_id(movie_id)
        
        if user_encoded is None or movie_encoded is None:
            return None
        
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_encoded])
            movie_tensor = torch.LongTensor([movie_encoded])
            
            prediction = self.model(user_tensor, movie_tensor)
            return prediction.item()
    
    def get_user_recommendations(self, user_id, n_recommendations=10, filter_already_liked=True):
        """
        Get movie recommendations for a specific user.
        
        Args:
            user_id: The user ID to generate recommendations for
            n_recommendations: Number of recommendations to return
            filter_already_liked: Whether to exclude movies the user has already rated
            
        Returns:
            DataFrame with recommended movies and their details
        """
        user_encoded = self._encode_user_id(user_id)
        if user_encoded is None:
            print(f"User {user_id} not found in training data. Cannot generate recommendations.")
            return None
        
        # Get user's rated movies if filtering is enabled
        rated_movies = set()
        if filter_already_liked:
            train_df = pd.read_csv("Recommender/data/processed/train.csv")
            user_ratings = train_df[train_df['userId'] == user_id]
            rated_movies = set(user_ratings['movieId'].values)
        
        # Predict ratings for all movies
        predictions = []
        for movie_id in self.encoders['movie'].classes_:
            if filter_already_liked and movie_id in rated_movies:
                continue
                
            predicted_rating = self.predict_rating(user_id, movie_id)
            if predicted_rating is not None:
                predictions.append({
                    'movieId': movie_id,
                    'predicted_rating': predicted_rating
                })
        
        # Sort by predicted rating and get top recommendations
        predictions_df = pd.DataFrame(predictions)
        predictions_df = predictions_df.sort_values('predicted_rating', ascending=False)
        top_recommendations = predictions_df.head(n_recommendations)
        
        # Add movie details
        recommendations = top_recommendations.merge(self.movies_df, on='movieId', how='left')
        recommendations['rank'] = range(1, len(recommendations) + 1)
        
        return recommendations[['rank', 'title', 'genres', 'predicted_rating', 'movieId']]
    
    def get_similar_movies(self, movie_id, n_similar=10):
        """
        Find movies similar to a given movie using embedding similarity.
        
        Args:
            movie_id: The movie ID to find similar movies for
            n_similar: Number of similar movies to return
            
        Returns:
            DataFrame with similar movies and their details
        """
        movie_encoded = self._encode_movie_id(movie_id)
        if movie_encoded is None:
            print(f"Movie {movie_id} not found in training data.")
            return None
        
        # Get movie embedding
        with torch.no_grad():
            movie_tensor = torch.LongTensor([movie_encoded])
            movie_embedding = self.model.movie_embedding(movie_tensor)
        
        # Calculate similarities with all other movies
        similarities = []
        for other_movie_id in self.encoders['movie'].classes_:
            if other_movie_id == movie_id:
                continue
                
            other_movie_encoded = self._encode_movie_id(other_movie_id)
            if other_movie_encoded is not None:
                other_movie_tensor = torch.LongTensor([other_movie_encoded])
                other_embedding = self.model.movie_embedding(other_movie_tensor)
                
                # Cosine similarity
                similarity = torch.cosine_similarity(movie_embedding, other_embedding).item()
                similarities.append({
                    'movieId': other_movie_id,
                    'similarity_score': similarity
                })
        
        # Sort by similarity and get top similar movies
        similarities_df = pd.DataFrame(similarities)
        similarities_df = similarities_df.sort_values('similarity_score', ascending=False)
        top_similar = similarities_df.head(n_similar)
        
        # Add movie details
        similar_movies = top_similar.merge(self.movies_df, on='movieId', how='left')
        similar_movies['rank'] = range(1, len(similar_movies) + 1)
        
        return similar_movies[['rank', 'title', 'genres', 'similarity_score', 'movieId']]
    
    def get_user_history(self, user_id, n_movies=10):
        """
        Get the movies a user has already rated (from training data).
        
        Args:
            user_id: The user ID
            n_movies: Number of recent movies to show
            
        Returns:
            DataFrame with user's rating history
        """
        train_df = pd.read_csv("Recommender/data/processed/train.csv")
        user_history = train_df[train_df['userId'] == user_id].sort_values('timestamp', ascending=False)
        
        if len(user_history) == 0:
            print(f"No history found for user {user_id}")
            return None
        
        return user_history[['title', 'genres', 'rating', 'year']].head(n_movies)

def demo_neural_recommendations():
    """
    Demo function to show how to use the neural recommender.
    """
    print("=== Neural Movie Recommender Demo ===\n")
    
    # Initialize recommender
    recommender = NeuralMovieRecommender()
    
    # Example user ID (you can change this)
    user_id = 1
    
    print(f"Getting recommendations for user {user_id}...")
    recommendations = recommender.get_user_recommendations(user_id, n_recommendations=5)
    
    if recommendations is not None:
        print("\nTop 5 Movie Recommendations:")
        print(recommendations[['rank', 'title', 'predicted_rating']].to_string(index=False))
    
    print(f"\nGetting user history for user {user_id}...")
    history = recommender.get_user_history(user_id, n_movies=3)
    
    if history is not None:
        print("\nRecent User History:")
        print(history[['title', 'rating']].to_string(index=False))
    
    # Example movie similarity
    movie_id = 1
    print(f"\nFinding movies similar to movie ID {movie_id}...")
    similar_movies = recommender.get_similar_movies(movie_id, n_similar=3)
    
    if similar_movies is not None:
        print("\nSimilar Movies:")
        print(similar_movies[['rank', 'title', 'similarity_score']].to_string(index=False))

if __name__ == '__main__':
    demo_neural_recommendations() 