#!/usr/bin/env python3
"""
Script to set up MLflow and run neural network training.
This handles the MLflow configuration and training execution.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def setup_mlflow():
    """Set up MLflow tracking URI."""
    print("Setting up MLflow...")
    
    # Set MLflow tracking URI
    tracking_uri = "http://localhost:5000"
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    
    print(f"SUCCESS: MLflow tracking URI set to: {tracking_uri}")
    return tracking_uri

def start_mlflow_server():
    """Start MLflow tracking server."""
    print("Starting MLflow server...")
    
    try:
        # Start MLflow server in background
        server_process = subprocess.Popen(
            ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for server to start
        time.sleep(3)
        
        print("SUCCESS: MLflow server started on http://localhost:5000")
        return server_process
        
    except Exception as e:
        print(f"ERROR: Failed to start MLflow server: {e}")
        print("Make sure MLflow is installed: pip install mlflow")
        return None

def run_neural_training():
    """Run the neural network training."""
    print("Running neural network training...")
    
    # Check if we're in a virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        # We're in a virtual environment, use current Python
        python_executable = sys.executable
        print(f"SUCCESS: Using virtual environment Python: {python_executable}")
    else:
        # Try to find virtual environment Python
        possible_venv_paths = [
            Path(".venv_mlops/Scripts/python.exe"),  # New project root venv
            Path(".venv/Scripts/python.exe"),  # Old project root venv
            Path("Recommender/.venv/Scripts/python.exe"),  # Recommender subdir venv
        ]
        
        python_executable = None
        for venv_path in possible_venv_paths:
            if venv_path.exists():
                python_executable = str(venv_path)
                print(f"SUCCESS: Found virtual environment: {python_executable}")
                break
        
        if not python_executable:
            print("ERROR: Virtual environment not found")
            print("Please activate your virtual environment first")
            return False
    
    try:
        # Set environment variables to handle Unicode properly
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        env['PYTHONLEGACYWINDOWSSTDIO'] = 'utf-8'
        
        # Run the training script
        result = subprocess.run(
            [python_executable, "Recommender/src/train_neural.py"],
            check=True,
            capture_output=True,
            text=True,
            env=env
        )
        
        print("SUCCESS: Training completed successfully!")
        print(result.stdout)
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Training failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def open_mlflow_ui():
    """Open MLflow UI in browser."""
    print("Opening MLflow UI...")
    
    try:
        webbrowser.open("http://localhost:5000")
        print("SUCCESS: MLflow UI opened in browser")
    except Exception as e:
        print(f"WARNING: Could not open browser automatically: {e}")
        print("Please manually open: http://localhost:5000")

def main():
    """Main function to orchestrate the training process."""
    print("MLflow Neural Training Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("Recommender").exists():
        print("ERROR: Please run this script from the project root (MLOps directory)")
        print("Current directory:", os.getcwd())
        return
    
    # Set up MLflow
    setup_mlflow()
    
    # Start MLflow server
    server_process = start_mlflow_server()
    if not server_process:
        return
    
    try:
        # Run training
        success = run_neural_training()
        
        if success:
            # Open MLflow UI
            open_mlflow_ui()
            
            print("\nSUCCESS: Training completed! Check MLflow UI for results.")
            print("\nWhat to look for in MLflow:")
            print("   - Experiment: 'movielens_als_tuning_neural'")
            print("   - Runs: Each Optuna trial")
            print("   - Metrics: Training/validation loss curves")
            print("   - Parameters: Hyperparameter values")
            print("   - Artifacts: Trained model files")
        
    finally:
        # Clean up server
        if server_process:
            print("\nStopping MLflow server...")
            server_process.terminate()
            server_process.wait()
            print("SUCCESS: MLflow server stopped")

if __name__ == "__main__":
    main() 