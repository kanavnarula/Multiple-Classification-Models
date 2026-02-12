"""
Random Forest Model Training and Evaluation
"""

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_curve
)
from utils.preprocessing import encode_features, prepare_train_test_split
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_metrics_bar_chart, display_metrics_grid
)
import matplotlib.pyplot as plt
import numpy as np


@st.cache_data
def train_random_forest(df, test_size, random_state, n_estimators, max_depth, min_samples_split):
    """Train Random Forest model and return results"""
    
    # Encode categorical features
    df_encoded, label_encoders = encode_features(df)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test, target_col, X, y = prepare_train_test_split(
        df_encoded, test_size, random_state
    )
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Model': 'Random Forest',
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC Score': roc_auc_score(y_test, y_pred_proba),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'MCC Score': matthews_corrcoef(y_test, y_pred)
    }
    
    # Additional results
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    class_report = classification_report(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return {
        'model': model,
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr),
        'classification_report': class_report,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'feature_importance': feature_importance,
        'y_test': y_test,
        'X_test': X_test,
        'feature_names': list(X.columns),
        'n_trees': n_estimators,
        'max_depth': max_depth
    }


def plot_feature_importance(feature_importance, top_n=15):
    """Plot feature importance from Random Forest"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    top_features = feature_importance.head(top_n)
    
    ax.barh(range(len(top_features)), top_features['importance'], color='forestgreen')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance - Random Forest', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    # Add value labels
    for i, v in enumerate(top_features['importance']):
        ax.text(v, i, f' {v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def display_random_forest_results(results):
    """Display Random Forest model results"""
    
    st.header("Random Forest Classifier Results")
    st.markdown(f"**Model Type:** Ensemble Learning - Random Forest")
    st.markdown(f"**Description:** Ensemble of {results['n_trees']} decision trees with max depth {results['max_depth']}")
    
    # Metrics
    st.subheader("Performance Metrics")
    display_metrics_grid(results['metrics'])
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(
            results['confusion_matrix'],
            'Confusion Matrix - Random Forest',
            'Greens'
        )
        st.pyplot(fig)
        
        # Show confusion matrix values
        cm = results['confusion_matrix']
        st.write(f"**True Negatives (TN):** {cm[0,0]}")
        st.write(f"**False Positives (FP):** {cm[0,1]}")
        st.write(f"**False Negatives (FN):** {cm[1,0]}")
        st.write(f"**True Positives (TP):** {cm[1,1]}")
    
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr = results['roc_curve']
        fig = plot_roc_curve(
            fpr, tpr,
            results['metrics']['AUC Score'],
            'ROC Curve - Random Forest',
            'forestgreen'
        )
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Metrics bar chart
    st.subheader("Metrics Overview")
    fig = plot_metrics_bar_chart(
        results['metrics'],
        'Random Forest - Performance on Test Data'
    )
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("Feature Importance Analysis")
    st.markdown("Random Forest calculates feature importance based on how much each feature decreases impurity across all trees.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_feature_importance(results['feature_importance'], top_n=15)
        st.pyplot(fig)
    
    with col2:
        st.markdown("**Top 10 Features:**")
        for idx, row in results['feature_importance'].head(10).iterrows():
            st.write(f"{idx+1}. **{row['feature']}**: {row['importance']:.4f}")
    
    st.markdown("---")
    
    # Feature Importance Table
    with st.expander("Full Feature Importance Table"):
        st.dataframe(
            results['feature_importance'].style.background_gradient(
                subset=['importance'], 
                cmap='Greens'
            ),
            use_container_width=True,
            height=400
        )
    
    # Classification Report
    with st.expander("Detailed Classification Report"):
        st.text(results['classification_report'])
    
    return results
