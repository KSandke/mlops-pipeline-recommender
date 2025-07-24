import pandas as pd
import joblib
import os

class MovieRecommender:
    def __init__(self, model_path="Recommender/models"):
        """
        Initialize the recommender by loading the trained model and mappings.
        """
        print("Loading trained model and mappings...")
        
        # Load model artifacts
        self.model = joblib.load(os.path.join(model_path, 'als_model.joblib'))
        self.user_map = joblib.load(os.path.join(model_path, 'user_map.joblib'))
        self.movie_map = joblib.load(os.path.join(model_path, 'movie_map.joblib'))
        self.interaction_matrix = joblib.load(os.path.join(model_path, 'interaction_matrix.joblib'))
        
        # Load movie metadata for readable recommendations
        self.movies_df = pd.read_csv("Recommender/data/raw/movies.csv")
        
        print("Model loaded successfully!")
        print(f"- Users in training: {len(self.user_map)}")
        print(f"- Movies in training: {len(self.movie_map)}")

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
        if user_id not in self.user_map:
            print(f"User {user_id} not found in training data. Cannot generate recommendations.")
            return None
        
        # Get user index
        user_idx = self.user_map.get_loc(user_id)
        
        # Generate recommendations
        recommended_items, scores = self.model.recommend(
            user_idx, 
            self.interaction_matrix[user_idx],
            N=n_recommendations,
            filter_already_liked_items=filter_already_liked
        )
        
        # Convert movie indices back to movie IDs
        recommended_movie_ids = [self.movie_map[idx] for idx in recommended_items]
        
        # Create recommendations DataFrame with movie details
        recommendations = pd.DataFrame({
            'movieId': recommended_movie_ids,
            'recommendation_score': scores,
            'rank': range(1, len(recommended_movie_ids) + 1)
        })
        
        # Merge with movie metadata
        recommendations = recommendations.merge(self.movies_df, on='movieId', how='left')
        
        return recommendations[['rank', 'title', 'genres', 'recommendation_score', 'movieId']]

    def get_similar_movies(self, movie_id, n_similar=10):
        """
        Find movies similar to a given movie.
        
        Args:
            movie_id: The movie ID to find similar movies for
            n_similar: Number of similar movies to return
        
        Returns:
            DataFrame with similar movies and their details
        """
        if movie_id not in self.movie_map:
            print(f"Movie {movie_id} not found in training data.")
            return None
        
        # Get movie index
        movie_idx = self.movie_map.get_loc(movie_id)
        
        # Find similar movies
        similar_items, scores = self.model.similar_items(movie_idx, N=n_similar + 1)
        
        # Remove the input movie itself (first result)
        similar_items = similar_items[1:]
        scores = scores[1:]
        
        # Convert movie indices back to movie IDs
        similar_movie_ids = [self.movie_map[idx] for idx in similar_items]
        
        # Create similar movies DataFrame
        similar_movies = pd.DataFrame({
            'movieId': similar_movie_ids,
            'similarity_score': scores,
            'rank': range(1, len(similar_movie_ids) + 1)
        })
        
        # Merge with movie metadata
        similar_movies = similar_movies.merge(self.movies_df, on='movieId', how='left')
        
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
        # Load training data to get user history
        train_df = pd.read_csv("Recommender/data/processed/train.csv")
        user_history = train_df[train_df['userId'] == user_id].sort_values('timestamp', ascending=False)
        
        if len(user_history) == 0:
            print(f"No history found for user {user_id}")
            return None
        
        return user_history[['title', 'genres', 'rating', 'year']].head(n_movies)

def demo_recommendations():
    """
    Demo function to show how to use the recommender.
    """
    print("=== Movie Recommender Demo ===\n")
    
    # Initialize recommender
    recommender = MovieRecommender()
    
    # Get a sample user ID from the training data
    sample_user_id = recommender.user_map[0]  # First user in the dataset
    
    print(f"\n--- Recommendations for User {sample_user_id} ---")
    recommendations = recommender.get_user_recommendations(sample_user_id, n_recommendations=5)
    if recommendations is not None:
        print(recommendations.to_string(index=False))
    
    print(f"\n--- User {sample_user_id}'s Rating History ---")
    history = recommender.get_user_history(sample_user_id, n_movies=5)
    if history is not None:
        print(history.to_string(index=False))
    
    # Get a sample movie ID and find similar movies
    sample_movie_id = recommender.movie_map[0]  # First movie in the dataset
    movie_title = recommender.movies_df[recommender.movies_df['movieId'] == sample_movie_id]['title'].iloc[0]
    
    print(f"\n--- Movies Similar to '{movie_title}' ---")
    similar_movies = recommender.get_similar_movies(sample_movie_id, n_similar=5)
    if similar_movies is not None:
        print(similar_movies.to_string(index=False))

if __name__ == '__main__':
    demo_recommendations() 