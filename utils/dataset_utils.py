"""
Common Dataset Utilities for Mushroom Classification Project

This module provides shared functions for loading, preprocessing, and
exploring the Mushroom Classification dataset across all model notebooks.
"""

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import LabelEncoder

# Configure display and plotting defaults
def setup_environment():
    """Configure pandas display options and matplotlib style"""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.width', None)
    
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 6)
    
    print("Environment configured successfully!")


def download_dataset():
    """
    Download the Mushroom Classification dataset from Kaggle
    
    Returns:
        str: Path to the downloaded dataset directory
    """
    print("="*70)
    print("MUSHROOM CLASSIFICATION DATASET - KAGGLE")
    print("="*70)
    
    path = kagglehub.dataset_download("uciml/mushroom-classification")
    print(f"\nPath to dataset files: {path}")
    
    # List files in the directory
    print("\nFiles in dataset directory:")
    for file in os.listdir(path):
        print(f"  - {file}")
    
    return path


def load_dataset(path):
    """
    Load the mushroom dataset from CSV file
    
    Args:
        path (str): Path to the dataset directory
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    # Find CSV file
    csv_file = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break
    
    if csv_file:
        print(f"Loading dataset from: {csv_file}")
        df = pd.read_csv(csv_file)
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {df.shape}")
        return df
    else:
        raise FileNotFoundError("No CSV file found in the dataset directory")


def get_dataset_info(df):
    """
    Display basic information about the dataset
    
    Args:
        df (pd.DataFrame): The dataset
    """
    print("="*70)
    print("DATASET INFORMATION")
    print("="*70)
    print(f"\nDataset Shape: {df.shape}")
    print(f"  - Samples: {df.shape[0]}")
    print(f"  - Features: {df.shape[1]}")
    print(f"\nColumn Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")
    
    print(f"\nData Types:")
    print(df.dtypes)
    
    print(f"\nMissing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  No missing values found!")
    else:
        print(missing[missing > 0])
    
    print(f"\nTarget Variable: {df.columns[0]}")
    print(f"Classes: {df[df.columns[0]].unique()}")
    print(f"Class Distribution:")
    print(df[df.columns[0]].value_counts())


def preprocess_data(df):
    """
    Encode categorical features using LabelEncoder
    
    Args:
        df (pd.DataFrame): Raw dataset with categorical features
        
    Returns:
        tuple: (df_encoded, label_encoders, target_col)
            - df_encoded: DataFrame with encoded features
            - label_encoders: Dictionary of LabelEncoder objects
            - target_col: Name of the target column
    """
    print("="*70)
    print("DATA PREPROCESSING")
    print("="*70)
    
    # Identify target column
    target_col = df.columns[0]
    print(f"\nTarget Variable: {target_col}")
    print(f"Classes: {df[target_col].unique()}")
    
    # Create a copy of the dataframe
    df_encoded = df.copy()
    
    # Initialize label encoders
    label_encoders = {}
    
    # Encode all categorical features
    print("\nEncoding categorical features...")
    for column in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    print("\nAll features encoded successfully!")
    print(f"\nEncoded dataset shape: {df_encoded.shape}")
    
    return df_encoded, label_encoders, target_col


def get_full_dataset():
    """
    Convenience function to download, load, and get basic info in one call
    
    Returns:
        pd.DataFrame: The loaded dataset
    """
    path = download_dataset()
    df = load_dataset(path)
    get_dataset_info(df)
    return df


def display_sample_data(df, n=10):
    """
    Display sample rows from the dataset
    
    Args:
        df (pd.DataFrame): The dataset
        n (int): Number of rows to display
        
    Returns:
        pd.DataFrame: First n rows of the dataset
    """
    print("="*70)
    print(f"DATASET PREVIEW (First {n} rows)")
    print("="*70)
    return df.head(n)
