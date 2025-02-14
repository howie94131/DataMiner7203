import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import argparse
import sys
from scipy.stats import mode

def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print("Dataset loaded successfully.")
        return data
    except FileNotFoundError:
        print(f"Error: The file at {file_path} was not found.")
        sys.exit(1)

def impute_missing_values(df):
    class_0 = df[df['Target (Col 106)'] == 0]
    class_1 = df[df['Target (Col 106)'] == 1]
    
    # Impute missing values for numerical features (columns 1-103)
    for col in df.columns[0:103]:
        if df[col].isnull().sum() > 0:
            mean_class_0 = class_0[col].mean()
            mean_class_1 = class_1[col].mean()
            df.loc[(df['Target (Col 106)'] == 0) & (df[col].isnull()), col] = mean_class_0
            df.loc[(df['Target (Col 106)'] == 1) & (df[col].isnull()), col] = mean_class_1
    
    # Impute missing values for nominal features (columns 104 and 105)
    for col in df.columns[103:105]:
        if df[col].isnull().sum() > 0:
            # Calculate mode, and handle cases where it returns a scalar
            mode_class_0 = mode(class_0[col].dropna(), nan_policy='omit').mode
            mode_class_1 = mode(class_1[col].dropna(), nan_policy='omit').mode
            
            # If mode result is an array, take the first element; otherwise, use it directly
            mode_class_0 = mode_class_0[0] if isinstance(mode_class_0, np.ndarray) else mode_class_0
            mode_class_1 = mode_class_1[0] if isinstance(mode_class_1, np.ndarray) else mode_class_1
            
            # Fill missing values based on the class
            df.loc[(df['Target (Col 106)'] == 0) & (df[col].isnull()), col] = mode_class_0
            df.loc[(df['Target (Col 106)'] == 1) & (df[col].isnull()), col] = mode_class_1
    
    print("Missing values have been imputed.")
    return df

def detect_and_remove_outliers(df, k=5, threshold=2.5):
    # Select numerical features for outlier detection
    numerical_features = df.columns[0:103]  # Columns from 'Num (Col 1)' to 'Num (Col 103)'
    data_numerical = df[numerical_features]
    
    # Fit KNN to calculate distances to k-nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(data_numerical)
    distances, _ = nbrs.kneighbors(data_numerical)
    
    # Calculate average distance to k-nearest neighbors for each point
    avg_distances = distances.mean(axis=1)
    
    # Determine outliers based on threshold for distance
    outlier_indices = np.where(avg_distances > threshold * avg_distances.mean())[0]
    print(f"Detected {len(outlier_indices)} outliers. Removing them.")
    
    # Drop rows corresponding to outliers from the dataset
    df = df.drop(index=outlier_indices).reset_index(drop=True)
    return df

def min_max_normalize(df):
    # Min-Max normalization for numerical features (columns 1-103)
    for col in df.columns[0:103]:
        min_val = df[col].min()
        max_val = df[col].max()
        df[col] = (df[col] - min_val) / (max_val - min_val)
    
    print("Min-Max normalization applied.")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, preprocess, and save the dataset for the data mining project.")
    parser.add_argument("file_path", type=str, help="Path to the CSV file containing the dataset.")
    
    args = parser.parse_args()
    
    # Load data
    dataset = load_data(args.file_path)
    
    # Impute missing values
    dataset = impute_missing_values(dataset)
    
    # Detect and remove outliers
    dataset = detect_and_remove_outliers(dataset)
    
    # Apply Min-Max normalization
    dataset = min_max_normalize(dataset)
    
    # Save the processed data to /processed/data.csv
    output_path = "processed/data.csv"
    dataset.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}.")
