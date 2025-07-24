import pandas as pd

def analyze_validation_data(
    validation_file="Recommender/data/processed/validation.csv",
    movies_file="Recommender/data/processed/movies.csv",
):
    """
    Analyze the validation set to understand why users are being filtered out.
    """
    print("=== VALIDATION SET ANALYSIS ===\n")
    
    # Load validation data
    validation_df = pd.read_csv(validation_file)
    movies_df = pd.read_csv(movies_file)

    print("--- Validation Set Statistics ---")
    print(validation_df.describe())

    # Get user IDs from the validation set that are also in the user map
    if 'rating' in validation_df.columns:
        print("\n--- Ratings Distribution ---")
        print(validation_df['rating'].value_counts().sort_index())

    if 'movieId' in validation_df.columns and 'title' in movies_df.columns:
        print("\n--- Top 10 Most Rated Movies in Validation ---")
        movie_counts = validation_df['movieId'].value_counts()
        top_movies = movie_counts.head(10).index
        for movie_id in top_movies:
            print(f"Movie ID: {movie_id}, Title: {movies_df[movies_df['movieId'] == movie_id]['title'].iloc[0]}")
    
    # Users with high ratings (>=4.0)
    high_rated = validation_df[validation_df['rating'] >= 4.0]
    print(f"\nUsers with ratings >= 4.0: {high_rated['userId'].nunique()}")
    print(f"Percentage of users with high ratings: {high_rated['userId'].nunique() / validation_df['userId'].nunique() * 100:.1f}%")
    
    # Rating distribution by percentage
    rating_dist = validation_df['rating'].value_counts(normalize=True).sort_index()
    print("\nRating distribution (percentages):")
    for rating, pct in rating_dist.items():
        print(f"Rating {rating}: {pct*100:.1f}%")
    
    # Check if each user has exactly one validation entry
    user_counts = validation_df['userId'].value_counts()
    print("\nEntries per user in validation set:")
    print(f"Users with 1 entry: {(user_counts == 1).sum()}")
    if (user_counts > 1).any():
        print(f"Users with >1 entry: {(user_counts > 1).sum()}")
    
    if (user_counts > 1).sum() > 0:
        print(f"Max entries per user: {user_counts.max()}")
    
    return validation_df

if __name__ == '__main__':
    analyze_validation_data() 