"""
Model Comparison Utilities
Compare multiple models side by side
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.visualization import plot_confusion_matrix


def display_comparison_results(lr_results, dt_results, knn_results=None):
    """Display comparison between two or three models"""
    
    st.success("All models trained successfully!")
    
    # Model Comparison Header
    st.header("Model Comparison")

    # Side-by-side metrics
    col1, col2, col3 = st.columns(3)
    
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
    
    if knn_results is not None:
        with col3:
            st.subheader("K-Nearest Neighbors")
            for metric, value in knn_results['metrics'].items():
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
    
    if knn_results is not None:
        comparison_df['K-Nearest Neighbors'] = [v for k, v in knn_results['metrics'].items() if k != 'Model']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(comparison_df))
    width = 0.25
    
    bars1 = ax.bar(x - width, comparison_df['Logistic Regression'], width, 
                   label='Logistic Regression', color='steelblue', edgecolor='black')
    bars2 = ax.bar(x, comparison_df['Decision Tree'], width, 
                   label='Decision Tree', color='forestgreen', edgecolor='black')
    
    if knn_results is not None:
        bars3 = ax.bar(x + width, comparison_df['K-Nearest Neighbors'], width, 
                       label='K-Nearest Neighbors', color='mediumpurple', edgecolor='black')
    else:
        bars3 = []
    
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
    for bars in [bars1, bars2] + ([bars3] if bars3 else []):
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Logistic Regression")
        fig = plot_confusion_matrix(
            lr_results['confusion_matrix'],
            'Confusion Matrix - Logistic Regression',
            'Blues'
        )
        st.pyplot(fig)
    
    with col2:
        st.subheader("Decision Tree")
        fig = plot_confusion_matrix(
            dt_results['confusion_matrix'],
            'Confusion Matrix - Decision Tree',
            'Greens'
        )
        st.pyplot(fig)
    
    if knn_results is not None:
        with col3:
            st.subheader("K-Nearest Neighbors")
            fig = plot_confusion_matrix(
                knn_results['confusion_matrix'],
                'Confusion Matrix - K-Nearest Neighbors',
                'Purples'
            )
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
    
    if knn_results is not None:
        fpr_knn, tpr_knn = knn_results['roc_data']
        ax.plot(fpr_knn, tpr_knn, color='purple', lw=2, 
               label=f'K-Nearest Neighbors (AUC = {knn_results["metrics"]["AUC Score"]:.4f})')
    
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
    
    if knn_results is not None:
        knn_score = knn_results['metrics']['Accuracy']
    
        if lr_score > dt_score and lr_score > knn_score:
            winner = "Logistic Regression"
            winner_score = lr_score
        elif dt_score > lr_score and dt_score > knn_score:
            winner = "Decision Tree"
            winner_score = dt_score
        elif knn_score > lr_score and knn_score > dt_score:
            winner = "K-Nearest Neighbors"
            winner_score = knn_score
        else:
            winner = "Tie"
            winner_score = lr_score  # All are equal
    
    else:
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
        st.success(f"🏆 **Winner: {winner}** with accuracy of **{winner_score:.4f}** ({winner_score*100:.2f}%)")
    else:
        st.info(f"🤝 **It's a Tie!** All models achieved **{winner_score:.4f}** ({winner_score*100:.2f}%) accuracy")
    
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
