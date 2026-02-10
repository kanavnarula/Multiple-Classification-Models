"""
Mushroom Classification - Streamlit Web Application
This app trains and evaluates a Logistic Regression model on the Mushroom Classification dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    precision_score, 
    recall_score, 
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
    roc_curve
)
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Mushroom Classification",
    page_icon="🍄",
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
st.title("🍄 Mushroom Classification with Logistic Regression")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)
    max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
    
    st.markdown("---")
    st.header("📊 About Dataset")
    st.info("""
    **Mushroom Classification Dataset**
    - Source: UCI Repository (Kaggle)
    - Task: Binary Classification
    - Classes: Edible / Poisonous
    - Features: 22 categorical attributes
    """)

# Cache dataset loading
@st.cache_data
def load_dataset():
    """Load and preprocess the mushroom dataset"""
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

# Train model function
@st.cache_data
def train_model(df, test_size, random_state, max_iter):
    """Train Logistic Regression model and return results"""
    
    # Encode categorical features
    df_encoded = df.copy()
    label_encoders = {}
    
    for column in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    # Separate features and target
    target_col = df.columns[0]
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Train model
    model = LogisticRegression(max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC Score': matthews_corrcoef(y_test, y_pred)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    # Classification report
    report = classification_report(y_test, y_pred, 
                                   target_names=['Edible', 'Poisonous'],
                                   output_dict=True)
    
    return {
        'model': model,
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr),
        'report': report,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'target_col': target_col
    }

# Main app
def main():
    # Load dataset
    df, path = load_dataset()
    
    if df is None:
        st.error("❌ Failed to load dataset. Please try again.")
        return
    
    st.success(f"✅ Dataset loaded successfully from: `{path}`")
    
    # Dataset Overview
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 1)
    with col3:
        st.metric("Classes", df[df.columns[0]].nunique())
    with col4:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Show dataset preview
    with st.expander("📋 View Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Class distribution
    with st.expander("📈 Class Distribution"):
        fig, ax = plt.subplots(figsize=(10, 5))
        df[df.columns[0]].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], 
                                               edgecolor='black', ax=ax)
        ax.set_title('Distribution of Mushroom Classes', fontsize=14, fontweight='bold')
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Train model button
    if st.button("🚀 Train Logistic Regression Model", type="primary", use_container_width=True):
        with st.spinner("Training model... Please wait."):
            results = train_model(df, test_size, random_state, max_iter)
        
        st.success("✅ Model trained successfully!")
        
        # Display metrics
        st.header("📊 Model Performance Metrics")
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        col4, col5, col6 = st.columns(3)
        
        metrics = results['metrics']
        
        with col1:
            st.metric("Accuracy", f"{metrics['Accuracy']:.4f}", 
                     f"{metrics['Accuracy']*100:.2f}%")
        with col2:
            st.metric("AUC Score", f"{metrics['AUC Score']:.4f}")
        with col3:
            st.metric("Precision", f"{metrics['Precision']:.4f}")
        with col4:
            st.metric("Recall", f"{metrics['Recall']:.4f}")
        with col5:
            st.metric("F1 Score", f"{metrics['F1 Score']:.4f}")
        with col6:
            st.metric("MCC Score", f"{metrics['MCC Score']:.4f}")
        
        st.markdown("---")
        
        # Visualizations
        st.header("📈 Model Visualizations")
        
        # Two columns for confusion matrix and ROC curve
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                       cmap='Blues', cbar=True,
                       xticklabels=['Edible', 'Poisonous'],
                       yticklabels=['Edible', 'Poisonous'], ax=ax)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show confusion matrix values
            cm = results['confusion_matrix']
            st.write(f"**True Negatives:** {cm[0,0]}")
            st.write(f"**False Positives:** {cm[0,1]}")
            st.write(f"**False Negatives:** {cm[1,0]}")
            st.write(f"**True Positives:** {cm[1,1]}")
        
        with col2:
            st.subheader("ROC Curve")
            fpr, tpr = results['roc_data']
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, 
                   label=f'ROC curve (AUC = {metrics["AUC Score"]:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                   label='Random Classifier')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("---")
        
        # Classification Report
        st.header("📄 Detailed Classification Report")
        report_df = pd.DataFrame(results['report']).transpose()
        st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), 
                    use_container_width=True)
        
        st.markdown("---")
        
        # Metrics Comparison Bar Chart
        st.header("📊 Metrics Comparison")
        fig, ax = plt.subplots(figsize=(12, 6))
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax.bar(metric_names, metric_values, color='steelblue', 
                     edgecolor='black', alpha=0.7)
        ax.set_ylim([0, 1.1])
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.5, label='Excellent (0.95)')
        ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.5, label='Good (0.80)')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("---")
        
        # Model Summary
        st.header("📋 Model Summary")
        st.info(f"""
        **Model:** Logistic Regression  
        **Training Samples:** {results['X_train'].shape[0]}  
        **Testing Samples:** {results['X_test'].shape[0]}  
        **Features:** {results['X_train'].shape[1]}  
        **Max Iterations:** {max_iter}  
        **Random State:** {random_state}  
        
        **Performance:** {'✅ EXCELLENT' if metrics['Accuracy'] >= 0.95 else '✅ VERY GOOD' if metrics['Accuracy'] >= 0.90 else '⚠️ GOOD'}
        """)
        
        # Download results
        st.header("💾 Download Results")
        
        # Create metrics dataframe
        metrics_df = pd.DataFrame([metrics])
        csv = metrics_df.to_csv(index=False)
        
        st.download_button(
            label="📥 Download Metrics as CSV",
            data=csv,
            file_name="mushroom_classification_metrics.csv",
            mime="text/csv",
            use_container_width=True
        )

if __name__ == "__main__":
    main()
