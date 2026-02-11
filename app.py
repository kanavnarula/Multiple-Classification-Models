"""
Mushroom Classification - Streamlit Web Application
This app trains and evaluates multiple classification models on the Mushroom Classification dataset

Architecture:
- utils/data_loader.py: Data loading and dataset info display
- utils/preprocessing.py: Data encoding and train-test split
- utils/visualization.py: Common plotting functions
- model/logistic_regression.py: Logistic Regression training and display
- model/decision_tree.py: Decision Tree training and display
- model/knn.py: K-Nearest Neighbors training and display
- model/comparison.py: Model comparison functionality
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils.data_loader import load_kaggle_dataset, load_uploaded_file, display_dataset_info
from utils.visualization import plot_class_distribution

# Import model modules
from model.logistic_regression import train_logistic_regression, display_logistic_regression_results
from model.decision_tree import train_decision_tree, display_decision_tree_results
from model.knn import train_knn, display_knn_results
from model.comparison import display_comparison_results

# Page configuration
st.set_page_config(
    page_title="Mushroom Classification",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Mushroom Classification - Multiple Models")
st.markdown("### Compare Multiple Classification Models")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Dataset Source")
    data_source = st.radio(
        "Choose Data Source",
        ["Default Dataset (Kaggle)", "Upload CSV File"],
        help="Upload your own CSV file or use the default mushroom dataset"
    )
    
    uploaded_file = None
    if data_source == "Upload CSV File":
        st.info("Upload smaller datasets for better performance on free tier.")
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=['csv'],
            help="Upload a CSV file with the same structure as the mushroom dataset. First column should be the target variable."
        )
    
    st.markdown("---")
    st.header("Model Selection")
    model_choice = st.selectbox(
        "Choose Classification Model",
        ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Compare All Models"]
    )
    
    st.markdown("---")
    st.header("Configuration")
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    # Model-specific parameters
    if model_choice == "Logistic Regression" or model_choice == "Compare All Models":
        st.subheader("Logistic Regression")
        max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
    
    if model_choice == "Decision Tree" or model_choice == "Compare All Models":
        st.subheader("Decision Tree")
        max_depth = st.slider("Max Depth", 3, 30, 10, 1)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
    
    if model_choice == "K-Nearest Neighbors" or model_choice == "Compare All Models":
        st.subheader("K-Nearest Neighbors")
        n_neighbors = st.slider("Number of Neighbors", 1, 20, 5, 1)
        weights = st.selectbox("Weights", ["uniform", "distance"], index=0)
    
    st.markdown("---")
    st.header("About Dataset")
    st.info("""
    **Mushroom Classification Dataset**
    - Source: UCI Repository (Kaggle)
    - Task: Binary Classification
    - Classes: Edible / Poisonous
    - Features: 22 categorical attributes
    - Total Samples: 8,124
    """)


# Main app
def main():
    # Load dataset based on source
    df = None
    
    if data_source == "Upload CSV File":
        if uploaded_file is not None:
            with st.spinner("Loading uploaded CSV file..."):
                df = load_uploaded_file(uploaded_file)
            
            if df is not None:
                st.success("CSV file uploaded and loaded successfully!")
            else:
                st.warning("Please upload a valid CSV file to proceed.")
                return
        else:
            st.info("👆 Please upload a CSV file from the sidebar to get started.")
            st.markdown("""
            ### Expected CSV Format:
            - **First column:** Target variable (class labels)
            - **Remaining columns:** Features
            - All columns should contain categorical or numerical data
            - **Example:** Mushroom dataset has 23 columns (1 target + 22 features)
            
            ### Tips:
            - Ensure your CSV has headers
            - First column will be treated as the target variable
            - Missing values will be displayed in the overview
            """)
            return
    else:
        # Load default dataset
        with st.spinner("Loading dataset from Kaggle..."):
            df, path = load_kaggle_dataset()
        
        if df is None:
            st.error("Failed to load dataset. Please try again.")
            return
        
        st.success("Dataset loaded successfully from Kaggle!")
    
    # Display dataset information
    display_dataset_info(df)
    
    # Show dataset preview
    with st.expander("View Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Class distribution
    with st.expander("Class Distribution"):
        fig = plot_class_distribution(df)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Train model button
    button_text = f"Train {model_choice}" if model_choice != "Compare All Models" else "Train All Models"

    if st.button(button_text, type="primary", use_container_width=True):
        
        if model_choice == "Logistic Regression":
            with st.spinner("Training Logistic Regression model..."):
                results = train_logistic_regression(df, test_size, random_state, max_iter)
            display_logistic_regression_results(results)
        
        elif model_choice == "Decision Tree":
            with st.spinner("Training Decision Tree model..."):
                results = train_decision_tree(df, test_size, random_state, max_depth, min_samples_split)
            display_decision_tree_results(results)
        
        elif model_choice == "K-Nearest Neighbors":
            with st.spinner("Training K-Nearest Neighbors model..."):
                results = train_knn(df, test_size, random_state, n_neighbors, weights)
            display_knn_results(results)
        
        else:  # Compare All Models
            with st.spinner("Training all models... Please wait."):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with st.spinner("Training Logistic Regression..."):
                        lr_results = train_logistic_regression(df, test_size, random_state, max_iter)
                    st.success("Logistic Regression Done")

                with col2:
                    with st.spinner("Training Decision Tree..."):
                        dt_results = train_decision_tree(df, test_size, random_state, max_depth, min_samples_split)
                    st.success("Decision Tree Done")
                
                with col3:
                    with st.spinner("Training KNN..."):
                        knn_results = train_knn(df, test_size, random_state, n_neighbors, weights)
                    st.success("KNN Done")
            
            display_comparison_results(lr_results, dt_results, knn_results)


if __name__ == "__main__":
    main()
