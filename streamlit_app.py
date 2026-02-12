"""
Mushroom Classification - Prediction App
Upload test data and get predictions from pre-trained models

Architecture:
- Pre-trained models are loaded from saved_models/ directory
- User uploads test.csv file
- App makes predictions and shows results
- If test file includes labels, shows evaluation metrics
"""

import streamlit as st
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import utilities
from utils.model_utils import (
    load_saved_models, load_label_encoders, load_feature_columns,
    preprocess_test_data, make_predictions, calculate_metrics, get_prediction_label
)
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, plot_metrics_bar_chart,
    display_metrics_grid
)

# Page configuration
st.set_page_config(
    page_title="Mushroom Classification - Prediction",
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
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .edible {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .poisonous {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.title("Mushroom Classification - Prediction System")
st.markdown("### Upload test data and get predictions from pre-trained models")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("🤖 Model Selection")
    
    # Load models
    models = load_saved_models()
    
    if models is not None:
        st.success(f"Loaded {len(models)} pre-trained models")
        
        model_choice = st.selectbox(
            "Choose Model for Prediction",
            ["Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Compare All Models"]
        )
    else:
        st.error("Please train models first!")
        st.code("python train_models.py", language="bash")
        st.stop()
    
    st.markdown("---")
    st.header("Upload Test Data")
    uploaded_file = st.file_uploader(
        "Upload Test CSV File",
        type=['csv'],
        help="Upload a CSV file with mushroom features. Can include target column for evaluation."
    )
    
    st.markdown("---")
    st.header("About")
    st.info("""
    **How to use:**
    1. Upload a test CSV file
    2. Select a model
    3. Click "Predict"
    4. View predictions and metrics
    
    **CSV Format:**
    - Can include target column (for evaluation)
    - Or just features (for prediction only)
    - Same structure as training data
    """)
    
    st.markdown("---")
    st.header("Model Info")
    if model_choice == "Logistic Regression":
        st.info("""
        **Logistic Regression**
        - Fast predictions
        - Good for linearly separable data
        - Probabilistic output
        """)
    elif model_choice == "Decision Tree":
        st.info("""
        **Decision Tree**
        - Interpretable rules
        - Handles non-linear patterns
        - Feature importance available
        """)
    elif model_choice == "K-Nearest Neighbors":
        st.info("""
        **K-Nearest Neighbors**
        - Instance-based learning
        - No training phase
        - K=5, Uniform weights
        """)


def display_predictions(df, predictions, probabilities, label_encoders, model_name="model"):
    """Display prediction results in a nice format"""
    st.header("Prediction Results")
    
    # Create results dataframe
    results_df = df.copy()
    results_df['Prediction'] = [get_prediction_label(p, label_encoders) for p in predictions]
    results_df['Confidence (Edible)'] = probabilities[:, 0]
    results_df['Confidence (Poisonous)'] = probabilities[:, 1]
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    
    edible_count = sum(predictions == 0)
    poisonous_count = sum(predictions == 1)
    
    with col1:
        st.metric("Total Predictions", len(predictions))
    with col2:
        st.metric("Predicted Edible", edible_count, f"{edible_count/len(predictions)*100:.1f}%")
    with col3:
        st.metric("Predicted Poisonous", poisonous_count, f"{poisonous_count/len(predictions)*100:.1f}%")
    
    st.markdown("---")
    
    # Show predictions table
    st.subheader("Detailed Predictions")
    
    # Add color coding
    def highlight_prediction(row):
        if row['Prediction'] == 'e':
            return ['background-color: #d4edda'] * len(row)
        else:
            return ['background-color: #f8d7da'] * len(row)
    
    st.dataframe(
        results_df.style.apply(highlight_prediction, axis=1),
        use_container_width=True,
        height=400
    )
    
    # Download predictions
    st.markdown("---")
    st.subheader("Download Predictions")
    csv = results_df.to_csv(index=False)
    
    # Generate unique key for download button
    button_key = f"download_{model_name.lower().replace(' ', '_').replace('-', '_')}"
    
    st.download_button(
        label="Download Predictions as CSV",
        data=csv,
        file_name=f"mushroom_predictions_{model_name.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        use_container_width=True,
        key=button_key
    )


def display_evaluation(metrics, cm, roc_data, model_name):
    """Display evaluation metrics"""
    st.header("Model Evaluation")
    
    # Display metrics grid
    st.subheader("Performance Metrics")
    display_metrics_grid(metrics)
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Confusion Matrix")
        color_map = 'Blues' if model_name == 'Logistic Regression' else ('Greens' if model_name == 'Decision Tree' else 'Purples')
        fig = plot_confusion_matrix(cm, f'Confusion Matrix - {model_name}', color_map)
        st.pyplot(fig)
        
        # Show values
        st.write(f"**True Negatives:** {cm[0,0]}")
        st.write(f"**False Positives:** {cm[0,1]}")
        st.write(f"**False Negatives:** {cm[1,0]}")
        st.write(f"**True Positives:** {cm[1,1]}")
    
    with col2:
        st.subheader("ROC Curve")
        fpr, tpr = roc_data
        color = 'darkorange' if model_name == 'Logistic Regression' else ('green' if model_name == 'Decision Tree' else 'purple')
        fig = plot_roc_curve(fpr, tpr, metrics['AUC Score'], f'ROC Curve - {model_name}', color)
        st.pyplot(fig)
    
    st.markdown("---")
    
    # Metrics bar chart
    st.subheader("Metrics Overview")
    fig = plot_metrics_bar_chart(metrics, f'{model_name} - Performance on Test Data')
    st.pyplot(fig)


def main():
    # Check if models are loaded
    if models is None:
        st.error("No models found. Please train models first by running: `python train_models.py`")
        return
    
    # Load encoders and feature columns
    label_encoders = load_label_encoders()
    feature_columns = load_feature_columns()
    
    if label_encoders is None or feature_columns is None:
        st.error("Missing model artifacts. Please run: `python train_models.py`")
        return
    
    # File upload section
    if uploaded_file is None:
        st.info("Please upload a test CSV file from the sidebar to get started.")
        
        st.markdown("""
        ### 💡 Tips
        - Load Test Data in Upload Test Data section.
        - Choose Model for Prediction from the dropdown.
        - Click on Predict.
        - It will take a few seconds to output the results.

        Note: Compare All mode will show predictions and metrics for all models side by side, but may take longer to run.
        """)
        return
    
    # Load test data
    try:
        df_test = pd.read_csv(uploaded_file)
        st.success(f"Test file loaded: {df_test.shape[0]} samples, {df_test.shape[1]} columns")
        
        # Show preview
        with st.expander("View Test Data Preview"):
            st.dataframe(df_test.head(10), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return
    
    st.markdown("---")
    
    # Predict button
    if st.button("Predict", type="primary", use_container_width=True):
        
        try:
            # Preprocess test data
            with st.spinner("Preprocessing test data..."):
                X_test, y_test, has_target = preprocess_test_data(df_test, label_encoders, feature_columns)
            
            if model_choice != "Compare All Models":
                # Single model prediction
                with st.spinner(f"Making predictions with {model_choice}..."):
                    model = models[model_choice]
                    predictions, probabilities = make_predictions(model, X_test)
                
                # Display predictions
                display_predictions(df_test, predictions, probabilities, label_encoders, model_choice)
                
                # If ground truth available, show evaluation
                if has_target and y_test is not None:
                    st.markdown("---")
                    metrics, cm, roc_data = calculate_metrics(y_test, predictions, probabilities)
                    display_evaluation(metrics, cm, roc_data, model_choice)
            
            else:
                # Compare all models
                st.header("Comparing All Models")
                
                all_predictions = {}
                all_metrics = {}
                
                # Make predictions with each model
                for model_name, model in models.items():
                    with st.spinner(f"Predicting with {model_name}..."):
                        predictions, probabilities = make_predictions(model, X_test)
                        all_predictions[model_name] = (predictions, probabilities)
                        
                        if has_target and y_test is not None:
                            metrics, cm, roc_data = calculate_metrics(y_test, predictions, probabilities)
                            all_metrics[model_name] = (metrics, cm, roc_data)
                
                # Display comparison
                if has_target and y_test is not None:
                    st.subheader("Model Performance Comparison")
                    
                    # Metrics comparison
                    comparison_df = pd.DataFrame({
                        model_name: metrics[0]
                        for model_name, metrics in all_metrics.items()
                    }).T
                    
                    st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
                    
                    # Find best model
                    best_model = comparison_df['Accuracy'].idxmax()
                    best_accuracy = comparison_df['Accuracy'].max()
                    st.success(f"**Best Model: {best_model}** with accuracy of **{best_accuracy:.4f}**")
                else:
                    st.warning("Ground truth not available. Showing predictions only.")
                
                # Show predictions for each model
                st.markdown("---")
                for model_name, (predictions, probabilities) in all_predictions.items():
                    with st.expander(f"📋 {model_name} Predictions"):
                        display_predictions(df_test, predictions, probabilities, label_encoders, model_name)
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.exception(e)


if __name__ == "__main__":
    main()
