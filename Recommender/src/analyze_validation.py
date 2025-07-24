import pandas as pd
import os

def analyze_validation_data():
    """
    Analyze the validation set to understand why users are being filtered out.
    """
    print("=== VALIDATION SET ANALYSIS ===\n")
    
    # Load validation data
    val_df = pd.read_csv("Recommender/data/processed/validation.csv")
    
    print(f"Total validation entries: {len(val_df)}")
    print(f"Unique users in validation: {val_df['userId'].nunique()}")
    print(f"Unique movies in validation: {val_df['movieId'].nunique()}")
    
    # Rating distribution
    print(f"\nRating distribution in validation set:")
    print(val_df['rating'].value_counts().sort_index())
    
    # Users with high ratings (>=4.0)
    high_rated = val_df[val_df['rating'] >= 4.0]
    print(f"\nUsers with ratings >= 4.0: {high_rated['userId'].nunique()}")
    print(f"Percentage of users with high ratings: {high_rated['userId'].nunique() / val_df['userId'].nunique() * 100:.1f}%")
    
    # Rating distribution by percentage
    rating_dist = val_df['rating'].value_counts(normalize=True).sort_index()
    print(f"\nRating distribution (percentages):")
    for rating, pct in rating_dist.items():
        print(f"Rating {rating}: {pct*100:.1f}%")
    
    # Check if each user has exactly one validation entry
    user_counts = val_df['userId'].value_counts()
    print(f"\nEntries per user in validation set:")
    print(f"Users with 1 entry: {(user_counts == 1).sum()}")
    print(f"Users with >1 entry: {(user_counts > 1).sum()}")
    
    if (user_counts > 1).sum() > 0:
        print(f"Max entries per user: {user_counts.max()}")
    
    return val_df

if __name__ == '__main__':
    analyze_validation_data() 