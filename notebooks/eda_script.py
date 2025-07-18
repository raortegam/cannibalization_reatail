import os
import pandas as pd
from pathlib import Path

def basic_eda(file_path, file_name):
    """
    Perform basic EDA on a CSV file and print the results.
    
    Args:
        file_path (str): Path to the CSV file
        file_name (str): Name of the file for display purposes
    """
    print(f"\n{'='*50}")
    print(f"ANALYZING: {file_name}")
    print(f"{'='*50}")
    
    try:
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        print(f"\nFile size: {file_size:.2f} MB")
        
        # Read the first few rows to get column names and data types
        df = pd.read_csv(file_path, nrows=5)
        
        # Display basic info
        print("\nFirst 5 rows:")
        display(df.head())
        
        # Read full file for dimensions
        df_full = pd.read_csv(file_path)
        
        # Display dimensions
        print(f"\nDimensions: {df_full.shape[0]} rows x {df_full.shape[1]} columns")
        
        # Display column names and data types
        print("\nColumn names and data types:")
        print(df_full.dtypes)
        
        # Basic statistics for numeric columns
        print("\nBasic statistics:")
        print(df_full.describe(include='all'))
        
        # Check for missing values
        print("\nMissing values per column:")
        print(df_full.isnull().sum())
        
    except Exception as e:
        print(f"Error analyzing {file_name}: {str(e)}")

def main():
    # Base directory
    base_dir = Path("../data/raw")
    
    # List of CSV files to analyze
    csv_files = [
        "holidays_events/holidays_events.csv",
        "items/items.csv",
        "oil/oil.csv",
        "sample_submission/sample_submission.csv",
        "stores/stores.csv",
        "test/test.csv",
        "train/train.csv",
        "transactions/transactions.csv"
    ]
    
    # Perform EDA for each file
    for csv_file in csv_files:
        file_path = base_dir / csv_file
        if file_path.exists():
            basic_eda(file_path, csv_file)
        else:
            print(f"\nFile not found: {file_path}")

if __name__ == "__main__":
    main()
