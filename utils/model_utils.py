"""
Model Prediction Utilities
Load saved models and make predictions on new data
"""

import pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, roc_curve
)


@st.cache_resource
def load_saved_models():
    """Load all pre-trained models from disk"""
    models = {}
    
    try:
        # Load Logistic Regression
        with open('saved_models/logistic_regression.pkl', 'rb') as f:
            models['Logistic Regression'] = pickle.load(f)
        
        # Load Decision Tree
        with open('saved_models/decision_tree.pkl', 'rb') as f:
            models['Decision Tree'] = pickle.load(f)
        
        # Load KNN
        with open('saved_models/knn.pkl', 'rb') as f:
            models['K-Nearest Neighbors'] = pickle.load(f)
        
        return models
    except FileNotFoundError:
        st.error("Models not found! Please run 'python train_models.py' first.")
        return None


@st.cache_resource
def load_label_encoders():
    """Load label encoders from disk"""
    try:
        with open('saved_models/label_encoders.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Label encoders not found! Please run 'python train_models.py' first.")
        return None


@st.cache_resource
def load_feature_columns():
    """Load feature columns order from disk"""
    try:
        with open('saved_models/feature_columns.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Feature columns not found! Please run 'python train_models.py' first.")
        return None


def preprocess_test_data(df, label_encoders, feature_columns):
    """
    Preprocess test data using saved label encoders
    
    Parameters:
    -----------
    df : DataFrame
        Test data to preprocess
    label_encoders : dict
        Saved label encoders
    feature_columns : list
        Expected feature column names
    
    Returns:
    --------
    DataFrame : Encoded test data
    """
    df_encoded = df.copy()
    
    # Check if target column exists
    target_col = list(label_encoders.keys())[0]
    has_target = target_col in df.columns
    
    # Encode all columns
    for column in df_encoded.columns:
        if column in label_encoders:
            le = label_encoders[column]
            try:
                # Handle unseen categories
                df_encoded[column] = df_encoded[column].apply(
                    lambda x: le.transform([x])[0] if x in le.classes_ else -1
                )
            except Exception as e:
                st.warning(f"Warning: Could not encode column '{column}': {str(e)}")
    
    # If has target, separate it
    if has_target:
        y_test = df_encoded[target_col]
        X_test = df_encoded[feature_columns]
        return X_test, y_test, has_target
    else:
        # No target column, all columns are features
        X_test = df_encoded[feature_columns] if all(col in df_encoded.columns for col in feature_columns) else df_encoded
        return X_test, None, has_target


def make_predictions(model, X_test):
    """
    Make predictions using a trained model
    
    Parameters:
    -----------
    model : sklearn model
        Trained model
    X_test : DataFrame
        Test features
    
    Returns:
    --------
    tuple : (predictions, probabilities)
    """
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    
    return predictions, probabilities


def calculate_metrics(y_test, y_pred, y_pred_proba):
    """
    Calculate evaluation metrics
    
    Parameters:
    -----------
    y_test : array
        True labels
    y_pred : array
        Predicted labels
    y_pred_proba : array
        Prediction probabilities
    
    Returns:
    --------
    dict : Dictionary of metrics
    """
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_pred_proba[:, 1]),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC Score': matthews_corrcoef(y_test, y_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba[:, 1])
    
    return metrics, cm, (fpr, tpr)


def get_prediction_label(prediction, label_encoders):
    """Convert prediction back to original label"""
    target_col = list(label_encoders.keys())[0]
    le = label_encoders[target_col]
    return le.inverse_transform([prediction])[0]
