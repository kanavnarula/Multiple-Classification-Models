"""
Visualization utilities for model evaluation
Common plotting functions used across all models
"""

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_confusion_matrix(cm, title, color_map='Blues'):
    """
    Plot confusion matrix heatmap
    
    Parameters:
    -----------
    cm : array
        Confusion matrix
    title : str
        Title for the plot
    color_map : str
        Color map for seaborn heatmap
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', 
               cmap=color_map, cbar=True,
               xticklabels=['Edible', 'Poisonous'],
               yticklabels=['Edible', 'Poisonous'], ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig


def plot_roc_curve(fpr, tpr, auc_score, title, color='darkorange'):
    """
    Plot ROC curve
    
    Parameters:
    -----------
    fpr : array
        False positive rate
    tpr : array
        True positive rate
    auc_score : float
        Area under the curve score
    title : str
        Title for the plot
    color : str
        Color for the ROC curve line
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color=color, lw=2, 
           label=f'ROC curve (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
           label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_metrics_bar_chart(metrics, title):
    """
    Plot horizontal bar chart for all metrics
    
    Parameters:
    -----------
    metrics : dict
        Dictionary of metric names and values
    title : str
        Title for the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    metric_names = [k for k in metrics.keys() if k != 'Model']
    metric_values = [v for k, v in metrics.items() if k != 'Model']
    
    bars = ax.bar(metric_names, metric_values, color='steelblue', 
                 edgecolor='black', alpha=0.7)
    ax.set_ylim([0, 1.1])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
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
    return fig


def display_metrics_grid(metrics):
    """Display metrics in a 2x3 grid using Streamlit"""
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
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


def plot_class_distribution(df):
    """Plot class distribution bar chart"""
    fig, ax = plt.subplots(figsize=(10, 5))
    df[df.columns[0]].value_counts().plot(kind='bar', color=['#2ecc71', '#e74c3c'], 
                                           edgecolor='black', ax=ax)
    ax.set_title('Distribution of Classes', fontsize=14, fontweight='bold')
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()
    return fig
