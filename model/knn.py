"""
K-Nearest Neighbors Model Training and Evaluation
"""

import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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
def train_knn(df, test_size, random_state, n_neighbors, weights):
    """Train K-Nearest Neighbors model and return results"""
    
    # Encode categorical features
    df_encoded, label_encoders = encode_features(df)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test, target_col, X, y = prepare_train_test_split(
        df_encoded, test_size, random_state
    )
    
    # Train model
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights,
        metric='minkowski',
        p=2
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'Model': 'K-Nearest Neighbors',
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
        'model_name': 'K-Nearest Neighbors',
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_data': (fpr, tpr),
        'report': report,
        'X_train': X_train,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'target_col': target_col,
        'feature_importance': None,
        'n_neighbors': n_neighbors,
        'weights': weights
    }


def display_knn_results(results):
    """Display results for K-Nearest Neighbors model"""
    
    st.success(f"{results['model_name']} trained successfully!")
    
    # Display metrics
    st.header("Model Performance Metrics")
    display_metrics_grid(results['metrics'])
    
    st.markdown("---")
    
    # Visualizations
    st.header("Model Visualizations")
    
    # Two columns for confusion matrix and ROC curve
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        fig = plot_confusion_matrix(
            results['confusion_matrix'], 
            'Confusion Matrix - K-Nearest Neighbors',
            'Purples'
        )
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
        fig = plot_roc_curve(
            fpr, tpr, 
            results['metrics']['AUC Score'],
            'ROC Curve - K-Nearest Neighbors',
            'purple'
        )
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Metrics Overview
    st.header("Metrics Overview")
    fig = plot_metrics_bar_chart(
        results['metrics'],
        'K-Nearest Neighbors - Performance Metrics'
    )
    st.pyplot(fig)
    
    st.markdown("---")
    
    # Model Summary
    st.header("Model Summary")
    summary_text = f"""
    **Model:** {results['model_name']}  
    **Training Samples:** {results['X_train'].shape[0]}  
    **Testing Samples:** {results['X_test'].shape[0]}  
    **Features:** {results['X_train'].shape[1]}  
    **Number of Neighbors:** {results['n_neighbors']}  
    **Weights:** {results['weights']}  
    **Distance Metric:** minkowski (p=2)  
    """
    
    perf_level = 'EXCELLENT' if results['metrics']['Accuracy'] >= 0.95 else 'VERY GOOD' if results['metrics']['Accuracy'] >= 0.90 else 'GOOD'
    summary_text += f"\n**Performance:** {perf_level}"
    
    st.info(summary_text)
    
    # Download results
    st.header("Download Results")
    metrics_df = pd.DataFrame([results['metrics']])
    csv = metrics_df.to_csv(index=False)
    
    st.download_button(
        label=f"Download {results['model_name']} Metrics as CSV",
        data=csv,
        file_name=f"mushroom_{results['model_name'].lower().replace(' ', '_')}_metrics.csv",
        mime="text/csv",
        use_container_width=True
    )
