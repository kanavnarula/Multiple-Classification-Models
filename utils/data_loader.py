"""
Data loading utilities for Mushroom Classification
Handles both Kaggle dataset download and CSV file uploads
"""

import streamlit as st
import pandas as pd
import kagglehub
import os


@st.cache_data
def load_kaggle_dataset():
    """Load and preprocess the mushroom dataset from Kaggle"""
    with st.spinner("Downloading dataset from Kaggle..."):
        path = kagglehub.dataset_download("uciml/mushroom-classification")
    
    # Find CSV file
    csv_file = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break
    
    if csv_file:
        df = pd.read_csv(csv_file)
        return df, path
    else:
        return None, None


@st.cache_data
def load_uploaded_file(uploaded_file):
    """Load CSV file uploaded by user"""
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None


def display_dataset_info(df):
    """Display dataset overview information"""
    st.header("Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Classes", df[df.columns[0]].nunique())
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
