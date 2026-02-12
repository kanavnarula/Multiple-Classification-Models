"""
Naive Bayes Model Training and Evaluation
"""

import streamlit as st
import pandas as pd
from sklearn.naive_bayes import GaussianNB
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


@st.cache_data
def train_naive_bayes(df, test_size, random_state):
    """Train Naive Bayes model and return results"""
    
    # Encode categorical features
    df_encoded, label_encoders = encode_features(df)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test, target_col, X, y = prepare_train_test_split(
        df_encoded, test_size, random_state
    )
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Model': 'Naive Bayes',
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
    
    # Class probabilities for visualization
    class_proba = pd.DataFrame(
        y_pred_proba,
        columns=['Probability']
    )
    
    return {
        'model': model,
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr),
        'classification_report': class_report,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'class_probabilities': class_proba,
        'y_test': y_test,
        'X_test': X_test,
        'feature_names': list(X.columns)
    }


## Visualization Area
def display_naive_bayes_results(results):
    """Display Naive Bayes model results"""
    
    st.header("Naive Bayes Classifier Results")
    st.markdown("**Model Type:** Gaussian Naive Bayes")
    st.markdown("**Description:** Probabilistic classifier based on Bayes' theorem with assumption of independence between features")
    
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
            'Confusion Matrix - Naive Bayes',
            'Oranges'
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
            'ROC Curve - Naive Bayes',
            'darkorange'
        )
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Metrics bar chart
    st.subheader("Metrics Overview")
    fig = plot_metrics_bar_chart(
        results['metrics'],
        'Naive Bayes - Performance on Test Data'
    )
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Classification Report
    with st.expander("Detailed Classification Report"):
        st.text(results['classification_report'])
    
    return results
