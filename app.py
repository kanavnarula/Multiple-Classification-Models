"""
Mushroom Classification - Streamlit Web Application
This app trains and evaluates multiple classification models on the Mushroom Classification dataset
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
from sklearn.tree import DecisionTreeClassifier
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
st.markdown("### Compare Multiple Models")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("Model Selection")
    model_choice = st.selectbox(
        "Choose Classification Model",
        ["Logistic Regression", "Decision Tree", "Compare Both Models"]
    )
    
    st.markdown("---")
    st.header("Configuration")
    test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random State", 0, 100, 42)
    
    # Model-specific parameters
    if model_choice == "Logistic Regression" or model_choice == "Compare Both Models":
        st.subheader("Logistic Regression")
        max_iter = st.slider("Max Iterations", 100, 2000, 1000, 100)
    
    if model_choice == "Decision Tree" or model_choice == "Compare Both Models":
        st.subheader("Decision Tree")
        max_depth = st.slider("Max Depth", 3, 30, 10, 1)
        min_samples_split = st.slider("Min Samples Split", 2, 20, 5, 1)
    
    st.markdown("---")
    st.header("About Dataset")
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
def train_logistic_regression(df, test_size, random_state, max_iter):
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
        'Model': 'Logistic Regression',
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
        'model_name': 'Logistic Regression',
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr),
        'report': report,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'target_col': target_col,
        'feature_importance': None
    }

@st.cache_data
def train_decision_tree(df, test_size, random_state, max_depth, min_samples_split):
    """Train Decision Tree model and return results"""
    
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
    model = DecisionTreeClassifier(
        max_depth=max_depth, 
        min_samples_split=min_samples_split,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Model': 'Decision Tree',
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
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    return {
        'model': model,
        'model_name': 'Decision Tree',
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr),
        'report': report,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'target_col': target_col,
        'feature_importance': feature_importance,
        'tree_depth': model.get_depth(),
        'n_leaves': model.get_n_leaves()
    }

def display_single_model_results(results):
    """Display results for a single model"""
    
    st.success(f"{results['model_name']} trained successfully!")
    
    # Display metrics
    st.header("Model Performance Metrics")
    
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
    st.header("Model Visualizations")
    
    # Two columns for confusion matrix and ROC curve
    col1, col2 = st.columns(2)
    
    color_map = 'Blues' if results['model_name'] == 'Logistic Regression' else 'Greens'
    roc_color = 'darkorange' if results['model_name'] == 'Logistic Regression' else 'green'
    
    with col1:
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   cmap=color_map, cbar=True,
                   xticklabels=['Edible', 'Poisonous'],
                   yticklabels=['Edible', 'Poisonous'], ax=ax)
        ax.set_title(f'Confusion Matrix - {results["model_name"]}', 
                    fontsize=14, fontweight='bold')
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
        ax.plot(fpr, tpr, color=roc_color, lw=2, 
               label=f'ROC curve (AUC = {metrics["AUC Score"]:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
               label='Random Classifier')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {results["model_name"]}', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Importance (for Decision Tree)
    if results['feature_importance'] is not None:
        st.header("Feature Importance")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig, ax = plt.subplots(figsize=(12, 8))
            top_features = results['feature_importance'].head(15)
            ax.barh(range(len(top_features)), top_features['Importance'], 
                   color='forestgreen', edgecolor='black')
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features['Feature'])
            ax.set_xlabel('Importance Score', fontsize=12)
            ax.set_ylabel('Features', fontsize=12)
            ax.set_title('Top 15 Feature Importances', fontsize=14, fontweight='bold')
            ax.invert_yaxis()
            ax.grid(axis='x', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.subheader("Tree Statistics")
            st.metric("Tree Depth", results['tree_depth'])
            st.metric("Number of Leaves", results['n_leaves'])
            
            st.subheader("Top 5 Features")
            for idx, row in results['feature_importance'].head(5).iterrows():
                st.write(f"**{row['Feature']}**: {row['Importance']:.4f}")
        
        st.markdown("---")
    
    # Classification Report
    st.header("📄 Detailed Classification Report")
    report_df = pd.DataFrame(results['report']).transpose()
    st.dataframe(report_df.style.highlight_max(axis=0, color='lightgreen'), 
                use_container_width=True)
    
    st.markdown("---")
    
    # Metrics Comparison Bar Chart
    st.header("Metrics Overview")
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_names = [k for k in metrics.keys() if k != 'Model']
    metric_values = [v for k, v in metrics.items() if k != 'Model']
    
    bars = ax.bar(metric_names, metric_values, color='steelblue', 
                 edgecolor='black', alpha=0.7)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{results["model_name"]} - Performance Metrics', 
                fontsize=14, fontweight='bold')
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
    st.header("Model Summary")
    summary_text = f"""
    **Model:** {results['model_name']}  
    **Training Samples:** {results['X_train'].shape[0]}  
    **Testing Samples:** {results['X_test'].shape[0]}  
    **Features:** {results['X_train'].shape[1]}  
    """
    
    if results['feature_importance'] is not None:
        summary_text += f"""**Max Depth:** {results['model'].max_depth}  
    **Min Samples Split:** {results['model'].min_samples_split}  
    **Actual Tree Depth:** {results['tree_depth']}  
    **Number of Leaves:** {results['n_leaves']}  
    """
    
    perf_level = 'EXCELLENT' if metrics['Accuracy'] >= 0.95 else 'VERY GOOD' if metrics['Accuracy'] >= 0.90 else 'GOOD'
    summary_text += f"\n**Performance:** {perf_level}"
    
    st.info(summary_text)
    
    # Download results
    st.header("Download Results")
    metrics_df = pd.DataFrame([metrics])
    csv = metrics_df.to_csv(index=False)
    
    st.download_button(
        label=f"Download {results['model_name']} Metrics as CSV",
        data=csv,
        file_name=f"mushroom_{results['model_name'].lower().replace(' ', '_')}_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )

def display_comparison_results(lr_results, dt_results):
    """Display comparison between two models"""
    
    st.success("Both models trained successfully!")
    
    # Model Comparison Header
    st.header("Model Comparison")

    # Side-by-side metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        for metric, value in lr_results['metrics'].items():
            if metric != 'Model':
                st.metric(metric, f"{value:.4f}")
    
    with col2:
        st.subheader("Decision Tree")
        for metric, value in dt_results['metrics'].items():
            if metric != 'Model':
                delta = value - lr_results['metrics'][metric]
                delta_str = f"{delta:+.4f}"
                st.metric(metric, f"{value:.4f}", delta=delta_str)
    
    st.markdown("---")
    
    # Metrics Comparison Chart
    st.header("Metrics Comparison Chart")
    
    comparison_df = pd.DataFrame({
        'Metric': [k for k in lr_results['metrics'].keys() if k != 'Model'],
        'Logistic Regression': [v for k, v in lr_results['metrics'].items() if k != 'Model'],
        'Decision Tree': [v for k, v in dt_results['metrics'].items() if k != 'Model']
    })
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, comparison_df['Logistic Regression'], width, 
                   label='Logistic Regression', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, comparison_df['Decision Tree'], width, 
                   label='Decision Tree', color='forestgreen', edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Metric'], rotation=45, ha='right')
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.axhline(y=0.95, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=0.80, color='orange', linestyle='--', alpha=0.3)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Side-by-side confusion matrices
    st.header("Confusion Matrices Comparison")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(lr_results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Blues', cbar=True,
                   xticklabels=['Edible', 'Poisonous'],
                   yticklabels=['Edible', 'Poisonous'], ax=ax)
        ax.set_title('Confusion Matrix - Logistic Regression', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Decision Tree")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(dt_results['confusion_matrix'], annot=True, fmt='d', 
                   cmap='Greens', cbar=True,
                   xticklabels=['Edible', 'Poisonous'],
                   yticklabels=['Edible', 'Poisonous'], ax=ax)
        ax.set_title('Confusion Matrix - Decision Tree', fontweight='bold')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown("---")
    
    # ROC Curves Comparison
    st.header("ROC Curves Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fpr_lr, tpr_lr = lr_results['roc_data']
    fpr_dt, tpr_dt = dt_results['roc_data']
    
    ax.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, 
           label=f'Logistic Regression (AUC = {lr_results["metrics"]["AUC Score"]:.4f})')
    ax.plot(fpr_dt, tpr_dt, color='green', lw=2, 
           label=f'Decision Tree (AUC = {dt_results["metrics"]["AUC Score"]:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
           label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Winner Declaration
    st.header("Best Model")
    
    lr_score = lr_results['metrics']['Accuracy']
    dt_score = dt_results['metrics']['Accuracy']
    
    if lr_score > dt_score:
        winner = "Logistic Regression"
        winner_score = lr_score
    elif dt_score > lr_score:
        winner = "Decision Tree"
        winner_score = dt_score
    else:
        winner = "Tie"
        winner_score = lr_score
    
    if winner != "Tie":
        st.success(f"{emoji} **Winner: {winner}** with accuracy of **{winner_score:.4f}** ({winner_score*100:.2f}%)")
    else:
        st.info(f"{emoji} **It's a Tie!** Both models achieved **{winner_score:.4f}** ({winner_score*100:.2f}%) accuracy")
    
    # Download comparison
    st.header("Download Comparison Results")
    csv = comparison_df.to_csv(index=False)
    
    st.download_button(
        label="Download Comparison as CSV",
        data=csv,
        file_name="mushroom_models_comparison.csv",
        mime="text/csv",
        use_container_width=True
    )

# Main app
def main():
    # Load dataset
    df, path = load_dataset()
    
    if df is None:
        st.error("Failed to load dataset. Please try again.")
        return
    
    st.success(f"Dataset loaded successfully from: `{path}`")
    
    # Dataset Overview
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
    
    # Show dataset preview
    with st.expander("View Dataset Preview"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Class distribution
    with st.expander("Class Distribution"):
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
    button_text = f"Train {model_choice}" if model_choice != "Compare Both Models" else "Train Both Models"
    
    if st.button(button_text, type="primary", use_container_width=True):
        
        if model_choice == "Logistic Regression":
            with st.spinner("Training Logistic Regression model..."):
                results = train_logistic_regression(df, test_size, random_state, max_iter)
            display_single_model_results(results)
        
        elif model_choice == "Decision Tree":
            with st.spinner("Training Decision Tree model..."):
                results = train_decision_tree(df, test_size, random_state, max_depth, min_samples_split)
            display_single_model_results(results)
        
        else:  # Compare Both Models
            with st.spinner("Training both models... Please wait."):
                lr_results = train_logistic_regression(df, test_size, random_state, max_iter)
                dt_results = train_decision_tree(df, test_size, random_state, max_depth, min_samples_split)
            
            display_comparison_results(lr_results, dt_results)

if __name__ == "__main__":
    main()
