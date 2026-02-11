"""
Decision Tree Model Training and Evaluation
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
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
def train_decision_tree(df, test_size, random_state, max_depth, min_samples_split):
    """Train Decision Tree model and return results"""
    
    # Encode categorical features
    df_encoded, label_encoders = encode_features(df)
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test, target_col, X, y = prepare_train_test_split(
        df_encoded, test_size, random_state
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


def plot_feature_importance(feature_importance):
    """Plot feature importance for Decision Tree"""
    fig, ax = plt.subplots(figsize=(12, 8))
    top_features = feature_importance.head(15)
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
    return fig


def display_decision_tree_results(results):
    """Display results for Decision Tree model"""
    
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
            'Confusion Matrix - Decision Tree',
            'Greens'
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
            'ROC Curve - Decision Tree',
            'green'
        )
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Feature Importance
    st.header("Feature Importance")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig = plot_feature_importance(results['feature_importance'])
        st.pyplot(fig)
    
    with col2:
        st.subheader("Tree Statistics")
        st.metric("Tree Depth", results['tree_depth'])
        st.metric("Number of Leaves", results['n_leaves'])
        
        st.subheader("Top 5 Features")
        for idx, row in results['feature_importance'].head(5).iterrows():
            st.write(f"**{row['Feature']}**: {row['Importance']:.4f}")
    
    st.markdown("---")
    
    # Metrics Overview
    st.header("Metrics Overview")
    fig = plot_metrics_bar_chart(
        results['metrics'],
        'Decision Tree - Performance Metrics'
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
    **Max Depth:** {results['model'].max_depth}  
    **Min Samples Split:** {results['model'].min_samples_split}  
    **Actual Tree Depth:** {results['tree_depth']}  
    **Number of Leaves:** {results['n_leaves']}  
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
