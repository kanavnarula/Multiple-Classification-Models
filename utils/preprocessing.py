"""
Data preprocessing utilities for Mushroom Classification
Handles encoding and train-test split
"""

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def encode_features(df):
    """Encode all categorical features using LabelEncoder"""
    df_encoded = df.copy()
    label_encoders = {}
    
    for column in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    return df_encoded, label_encoders


def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """
    Prepare features and target, then split into train and test sets
    
    Parameters:
    -----------
    df : DataFrame
        The dataframe to split (should be encoded)
    test_size : float
        Proportion of dataset to include in test split
    random_state : int
        Random state for reproducibility
    
    Returns:
    --------
    tuple : (X_train, X_test, y_train, y_test, target_col, X, y)
    """
    # Separate features and target (first column is assumed to be target)
    target_col = df.columns[0]
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, target_col, X, y
