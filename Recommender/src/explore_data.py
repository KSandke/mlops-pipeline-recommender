import pandas as pd
import os

def explore_data(raw_data_path="Recommender/data/raw"):
    """
    Loads and provides a basic exploration of the CSV files in the raw data folder.

    For each file, it prints:
    - First 5 rows (head)
    - Data types and non-null counts (info)
    - Descriptive statistics (describe)
    - Count of missing values per column
    """
    print(f"Exploring data in: {raw_data_path}")
    
    csv_files = [f for f in os.listdir(raw_data_path) if f.endswith('.csv')]
    
    for file_name in csv_files:
        file_path = os.path.join(raw_data_path, file_name)
        print(f"--- Exploring {file_name} ---")
        
        try:
            # For very large files, consider using pd.read_csv(..., nrows=...) 
            # or chunksize for initial exploration.
            df = pd.read_csv(file_path)
            
            print("\n--- Head ---")
            print(df.head())
            
            print("\n--- Info ---")
            df.info(verbose=False)
            
            print("\n--- Describe ---")
            print(df.describe())
            
            print("\n--- Missing Values ---")
            print(df.isnull().sum())
            
        except Exception as e:
            print(f"Could not process {file_name}. Error: {e}")
            
        print("\n" + "="*80 + "\n")

if __name__ == '__main__':
    # The script assumes it's run from the root of the MLOps project.
    # If your current working directory is different, you might need to adjust the path.
    explore_data() 