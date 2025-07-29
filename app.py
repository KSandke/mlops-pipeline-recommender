import gradio as gr
import joblib
import pandas as pd
import os

# --- Configuration ---
MODEL_PATH = "Recommender/models"
DATA_PATH = "Recommender/data/raw"
MODEL_ARTIFACTS = {
    "model": "als_model.joblib",
    "user_map": "user_map.joblib",
    "movie_map": "movie_map.joblib",
    "interaction_matrix": "interaction_matrix.joblib",
}
MOVIE_METADATA_FILE = "movies.csv"

# --- Load Artifacts ---
def load_artifacts(path):
    """Load a single artifact using joblib."""
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"Error: Artifact not found at {path}")
        return None
    except Exception as e:
        print(f"An error occurred while loading {path}: {e}")
        return None

# Load all model components
print("Loading model artifacts...")
artifacts = {
    key: load_artifacts(os.path.join(MODEL_PATH, filename))
    for key, filename in MODEL_ARTIFACTS.items()
}

# Load movie metadata
print("Loading movie metadata...")
movies_df = None
try:
    movies_df = pd.read_csv(os.path.join(DATA_PATH, MOVIE_METADATA_FILE))
    # Create a lookup dictionary for movie titles
    movie_titles = movies_df.set_index('movieId')['title'].to_dict()
except FileNotFoundError:
    print(f"Error: Movie metadata not found at {os.path.join(DATA_PATH, MOVIE_METADATA_FILE)}")
except Exception as e:
    print(f"An error occurred while loading movie metadata: {e}")

# --- Recommendation Logic ---
def get_recommendations(user_id, n_recs):
    """Generate movie recommendations for a given user ID."""
    # Check if all artifacts are loaded
    if any(v is None for v in artifacts.values()) or movies_df is None:
        return "Model or data files are missing. Please check the console for errors.", None

    model = artifacts["model"]
    user_map = artifacts["user_map"]
    movie_map = artifacts["movie_map"]
    interaction_matrix = artifacts["interaction_matrix"]

    # Validate User ID
    if user_id not in user_map:
        return f"User ID {user_id} not found. Please try a different ID.", None

    user_idx = user_map.get_loc(user_id)

    # Get recommendations
    try:
        ids, scores = model.recommend(
            user_idx,
            interaction_matrix[user_idx],
            N=int(n_recs),
            filter_already_liked_items=True,
        )
    except Exception as e:
        return f"An error occurred during recommendation: {e}", None

    # Format output
    recs_df = pd.DataFrame({
        "Movie ID": [movie_map[i] for i in ids],
        "Score": scores,
    })
    
    # Add movie titles
    recs_df["Title"] = recs_df["Movie ID"].map(movie_titles)
    
    # Reorder columns for better readability
    recs_df = recs_df[["Movie ID", "Title", "Score"]]

    return f"Top {len(recs_df)} recommendations for User {user_id}:", recs_df

# --- Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 Movie Recommendation System
        Enter a User ID to get personalized movie recommendations from our ALS model.
        """
    )

    with gr.Row():
        user_id_input = gr.Number(label="User ID", info="e.g., 5, 123, 500", value=5)
        n_recs_slider = gr.Slider(minimum=5, maximum=20, value=10, step=1, label="Number of Recommendations")

    submit_button = gr.Button("Get Recommendations", variant="primary")

    with gr.Column():
        status_output = gr.Textbox(label="Status", interactive=False)
        results_output = gr.DataFrame(
            headers=["Movie ID", "Title", "Score"],
            datatype=["number", "str", "number"],
            label="Recommendations"
        )

    submit_button.click(
        fn=get_recommendations,
        inputs=[user_id_input, n_recs_slider],
        outputs=[status_output, results_output]
    )
    
    gr.Examples(
        examples=[[10], [123], [500]],
        inputs=user_id_input,
    )

# --- Launch the App ---
if __name__ == "__main__":
    if all(v is not None for v in artifacts.values()) and movies_df is not None:
        print("All artifacts loaded successfully. Launching Gradio demo...")
        demo.launch()
    else:
        print("Could not launch Gradio demo due to missing files. Please check the errors above.") 