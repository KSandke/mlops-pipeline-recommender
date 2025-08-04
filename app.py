import gradio as gr
import joblib
import pandas as pd
import os
import scipy.sparse as sp

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

# --- Reverse Mapping ---
movie_id_map = None
if movies_df is not None:
    # Create a lookup for movie ID from title
    movie_id_map = movies_df.set_index("title")["movieId"].to_dict()

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


def get_new_user_recommendations(selected_movies, n_recs):
    """Generate recommendations for a new user based on selected movies."""
    if any(v is None for v in artifacts.values()) or movies_df is None or movie_id_map is None:
        return "Model or data files are missing. Please check the console for errors.", None

    model = artifacts["model"]
    movie_map = artifacts["movie_map"]  # This is movieId -> matrix index
    interaction_matrix = artifacts["interaction_matrix"]

    if not selected_movies:
        return "Please select at least one movie.", None

    # Create a new user vector
    new_user_vector = sp.csr_matrix(
        (1, interaction_matrix.shape[1]), dtype=interaction_matrix.dtype
    )

    # Get movie indices for selected movies
    selected_movie_ids = [movie_id_map[title] for title in selected_movies]

    movie_indices = []
    for m_id in selected_movie_ids:
        try:
            # movie_map is a pandas Index
            loc = movie_map.get_loc(m_id)
            movie_indices.append(loc)
        except KeyError:
            # Movie might not be in the model's vocabulary
            print(f"Warning: Movie ID {m_id} not found in model's movie map.")
            pass

    if not movie_indices:
        return (
            "None of the selected movies are in the model's training data. Please select different movies.",
            None,
        )

    # Set liked items in the new user vector
    new_user_vector[0, movie_indices] = 1

    # Get recommendations
    try:
        # For a new user, user_id is 0, and we pass the new vector
        ids, scores = model.recommend(
            0,
            new_user_vector,
            N=int(n_recs),
            filter_already_liked_items=True,
        )
    except Exception as e:
        return f"An error occurred during recommendation: {e}", None

    # Format output
    recs_df = pd.DataFrame(
        {
            "Movie ID": [movie_map[i] for i in ids],
            "Score": scores,
        }
    )

    # Add movie titles
    recs_df["Title"] = recs_df["Movie ID"].map(movie_titles)

    # Reorder columns
    recs_df = recs_df[["Movie ID", "Title", "Score"]]

    return f"Top {len(recs_df)} recommendations for you:", recs_df


# --- Gradio Interface ---
def create_recommendation_tab(is_new_user=False):
    """Creates a recommendation tab for either new or existing users."""
    if is_new_user:
        markdown_text = "New here? Select some movies you like to get instant recommendations!"
        all_movie_titles = sorted(list(movie_titles.values()))
        inputs = [
            gr.Dropdown(
                all_movie_titles,
                multiselect=True,
                label="Select movies you like",
                info="Start typing to search for movies.",
            ),
            gr.Slider(
                minimum=5, maximum=20, value=10, step=1, label="Number of Recommendations"
            ),
        ]
        button_text = "Get My Recommendations"
        rec_function = get_new_user_recommendations
        output_label = "Your Recommendations"
    else:
        markdown_text = "Enter a User ID to get personalized movie recommendations from our ALS model."
        user_id_input = gr.Number(label="User ID", info="e.g., 5, 123, 500", value=5)
        inputs = [
            user_id_input,
            gr.Slider(
                minimum=5, maximum=20, value=10, step=1, label="Number of Recommendations"
            ),
        ]
        button_text = "Get Recommendations"
        rec_function = get_recommendations
        output_label = "Recommendations"

    gr.Markdown(markdown_text)
    with gr.Row():
        for component in inputs:
            # This renders the component. In Gradio, defining a component in a scope renders it.
            pass

    submit_button = gr.Button(button_text, variant="primary")

    with gr.Column():
        status_output = gr.Textbox(label="Status", interactive=False)
        results_output = gr.DataFrame(
            headers=["Movie ID", "Title", "Score"],
            datatype=["number", "str", "number"],
            label=output_label,
        )

    submit_button.click(
        fn=rec_function,
        inputs=inputs,
        outputs=[status_output, results_output],
    )

    if not is_new_user:
        gr.Examples(
            examples=[[10], [123], [500]],
            inputs=inputs[0],  # user_id_input
        )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎬 Movie Recommendation System
        """
    )
    with gr.Tabs():
        with gr.TabItem("Recommend for Existing User"):
            create_recommendation_tab(is_new_user=False)
        with gr.TabItem("Recommend for New User"):
            create_recommendation_tab(is_new_user=True)


# --- Launch the App ---
if __name__ == "__main__":
    if all(v is not None for v in artifacts.values()) and movies_df is not None:
        print("All artifacts loaded successfully. Launching Gradio demo...")
        demo.launch()
    else:
        print(
            "Could not launch Gradio demo due to missing files. Please check the errors above."
        ) 