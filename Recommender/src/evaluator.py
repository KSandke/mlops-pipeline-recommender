import pandas as pd
import numpy as np
import joblib
import os
from collections import defaultdict
import yaml

class RecommenderEvaluator:
    def __init__(self, model_path="Recommender/models", config_path="Recommender/configs/model_config.yaml"):
        """
        Initialize the evaluator by loading the trained model and validation data.
        """
        print("Loading trained model and validation data...")
        
        # Load model artifacts
        self.model = joblib.load(os.path.join(model_path, 'als_model.joblib'))
        self.user_map = joblib.load(os.path.join(model_path, 'user_map.joblib'))
        self.movie_map = joblib.load(os.path.join(model_path, 'movie_map.joblib'))
        self.interaction_matrix = joblib.load(os.path.join(model_path, 'interaction_matrix.joblib'))
        
        # Load validation data
        self.validation_df = pd.read_csv("Recommender/data/processed/validation.csv")
        
        # Load config for evaluation parameters
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.eval_k = config.get('evaluation', {}).get('k', 10)
        
        print(f"Evaluator loaded successfully!")
        print(f"- Validation set size: {len(self.validation_df)}")
        print(f"- Evaluation k: {self.eval_k}")

    def precision_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate precision@k for a single user.
        
        Args:
            recommended_items: List of recommended item IDs (ordered by score)
            relevant_items: Set of relevant/liked item IDs
            k: Number of top recommendations to consider
        
        Returns:
            Precision@k value
        """
        if k == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_recommended = len(set(top_k_recommendations) & set(relevant_items))
        
        return relevant_recommended / k

    def recall_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate recall@k for a single user.
        
        Args:
            recommended_items: List of recommended item IDs (ordered by score)
            relevant_items: Set of relevant/liked item IDs
            k: Number of top recommendations to consider
        
        Returns:
            Recall@k value
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        relevant_recommended = len(set(top_k_recommendations) & set(relevant_items))
        
        return relevant_recommended / len(relevant_items)

    def mean_average_precision_at_k(self, recommended_items, relevant_items, k):
        """
        Calculate Mean Average Precision@k for a single user.
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k_recommendations = recommended_items[:k]
        score = 0.0
        num_hits = 0.0
        
        for i, item in enumerate(top_k_recommendations):
            if item in relevant_items:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        return score / min(len(relevant_items), k)

    def evaluate_user(self, user_id, k_values=[5, 10, 20]):
        """
        Evaluate recommendations for a single user.
        
        Args:
            user_id: User ID to evaluate
            k_values: List of k values to evaluate
        
        Returns:
            Dictionary with evaluation metrics
        """
        if user_id not in self.user_map:
            return None
        
        # Get user's validation items (items they actually liked)
        user_validation = self.validation_df[self.validation_df['userId'] == user_id]
        if len(user_validation) == 0:
            return None
        
        # Consider items with rating >= 4.0 as "relevant"
        relevant_items = set(user_validation[user_validation['rating'] >= 4.0]['movieId'].tolist())
        
        if len(relevant_items) == 0:
            return None
        
        # Generate recommendations
        user_idx = self.user_map.get_loc(user_id)
        recommended_items, scores = self.model.recommend(
            user_idx, 
            self.interaction_matrix[user_idx],
            N=max(k_values),
            filter_already_liked_items=True
        )
        
        # Convert movie indices back to movie IDs
        recommended_movie_ids = [self.movie_map[idx] for idx in recommended_items]
        
        # Calculate metrics for different k values
        metrics = {}
        for k in k_values:
            metrics[f'precision@{k}'] = self.precision_at_k(recommended_movie_ids, relevant_items, k)
            metrics[f'recall@{k}'] = self.recall_at_k(recommended_movie_ids, relevant_items, k)
            metrics[f'map@{k}'] = self.mean_average_precision_at_k(recommended_movie_ids, relevant_items, k)
        
        return metrics

    def evaluate_all_users(self, k_values=[5, 10, 20], sample_size=None):
        """
        Evaluate recommendations for all users in validation set.
        
        Args:
            k_values: List of k values to evaluate
            sample_size: If specified, evaluate only a random sample of users
        
        Returns:
            Dictionary with aggregated evaluation metrics
        """
        print("Starting evaluation on validation set...")
        
        # Get unique users from validation set
        validation_users = self.validation_df['userId'].unique()
        
        # Sample users if specified
        if sample_size and sample_size < len(validation_users):
            validation_users = np.random.choice(validation_users, size=sample_size, replace=False)
            print(f"Evaluating on sample of {sample_size} users...")
        else:
            print(f"Evaluating on all {len(validation_users)} users...")
        
        # Collect metrics for all users
        all_metrics = defaultdict(list)
        evaluated_users = 0
        
        for i, user_id in enumerate(validation_users):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(validation_users)} users...")
            
            user_metrics = self.evaluate_user(user_id, k_values)
            if user_metrics:
                for metric_name, value in user_metrics.items():
                    all_metrics[metric_name].append(value)
                evaluated_users += 1
        
        # Calculate average metrics
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[metric_name] = np.mean(values)
            avg_metrics[f'{metric_name}_std'] = np.std(values)
        
        avg_metrics['evaluated_users'] = evaluated_users
        avg_metrics['total_validation_users'] = len(validation_users)
        
        print(f"Evaluation completed on {evaluated_users} users.")
        return avg_metrics

    def generate_evaluation_report(self, k_values=[5, 10, 20], sample_size=1000):
        """
        Generate a comprehensive evaluation report.
        
        Args:
            k_values: List of k values to evaluate
            sample_size: Number of users to sample for evaluation
        
        Returns:
            Dictionary with evaluation results and saves to file
        """
        print("=== RECOMMENDATION SYSTEM EVALUATION ===\n")
        
        # Run evaluation
        metrics = self.evaluate_all_users(k_values=k_values, sample_size=sample_size)
        
        # Print results
        print("BASELINE PERFORMANCE METRICS:")
        print("-" * 40)
        for k in k_values:
            print(f"Precision@{k}: {metrics[f'precision@{k}']:.4f} (±{metrics[f'precision@{k}_std']:.4f})")
            print(f"Recall@{k}:    {metrics[f'recall@{k}']:.4f} (±{metrics[f'recall@{k}_std']:.4f})")
            print(f"MAP@{k}:       {metrics[f'map@{k}']:.4f} (±{metrics[f'map@{k}_std']:.4f})")
            print()
        
        print(f"Users evaluated: {metrics['evaluated_users']}/{metrics['total_validation_users']}")
        
        # Save results to file
        results_file = "Recommender/models/evaluation_results.txt"
        with open(results_file, 'w') as f:
            f.write("=== RECOMMENDATION SYSTEM BASELINE EVALUATION ===\n\n")
            f.write(f"Model: ALS Collaborative Filtering\n")
            f.write(f"Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Users Evaluated: {metrics['evaluated_users']}/{metrics['total_validation_users']}\n\n")
            f.write("PERFORMANCE METRICS:\n")
            f.write("-" * 40 + "\n")
            for k in k_values:
                f.write(f"Precision@{k}: {metrics[f'precision@{k}']:.4f} (±{metrics[f'precision@{k}_std']:.4f})\n")
                f.write(f"Recall@{k}:    {metrics[f'recall@{k}']:.4f} (±{metrics[f'recall@{k}_std']:.4f})\n")
                f.write(f"MAP@{k}:       {metrics[f'map@{k}']:.4f} (±{metrics[f'map@{k}_std']:.4f})\n\n")
            
            f.write("\nNOTES:\n")
            f.write("- Relevant items defined as ratings >= 4.0\n")
            f.write("- Metrics calculated on time-based validation split\n")
            f.write("- This serves as the baseline for future model improvements\n")
        
        print(f"\nResults saved to: {results_file}")
        
        return metrics

def main():
    """
    Run the evaluation and generate baseline performance report.
    """
    evaluator = RecommenderEvaluator()
    
    # Generate comprehensive evaluation report
    results = evaluator.generate_evaluation_report(
        k_values=[5, 10, 20], 
        sample_size=1000  # Evaluate on 1000 users for speed
    )
    
    return results

if __name__ == '__main__':
    main() 