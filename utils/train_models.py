"""
Model Training Script - Train models offline and save them
This script trains all three models on the full training dataset and saves them to disk
"""

import pandas as pd
import pickle
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef
)
import kagglehub

def load_data():
    """Load the mushroom dataset from Kaggle"""
    print("Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("uciml/mushroom-classification")
    
    # Find CSV file
    csv_file = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break
    
    if csv_file:
        df = pd.read_csv(csv_file)
        print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    else:
        raise FileNotFoundError("CSV file not found in dataset")

def preprocess_data(df):
    """Encode categorical features"""
    print("\nEncoding categorical features...")
    df_encoded = df.copy()
    label_encoders = {}
    
    for column in df_encoded.columns:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le
    
    print(f"Encoded {len(label_encoders)} features")
    return df_encoded, label_encoders

def train_models(X_train, y_train, X_test, y_test):
    """Train all three models"""
    models = {}
    metrics = {}
    
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # 1. Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
    
    metrics['Logistic Regression'] = {
        'Accuracy': accuracy_score(y_test, y_pred_lr),
        'AUC Score': roc_auc_score(y_test, y_pred_proba_lr),
        'Precision': precision_score(y_test, y_pred_lr),
        'Recall': recall_score(y_test, y_pred_lr),
        'F1 Score': f1_score(y_test, y_pred_lr),
        'MCC Score': matthews_corrcoef(y_test, y_pred_lr)
    }
    models['logistic_regression'] = lr_model
    print(f"   ✓ Accuracy: {metrics['Logistic Regression']['Accuracy']:.4f}")
    
    # 2. Decision Tree
    print("\n2. Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=5, random_state=42)
    dt_model.fit(X_train, y_train)
    y_pred_dt = dt_model.predict(X_test)
    y_pred_proba_dt = dt_model.predict_proba(X_test)[:, 1]
    
    metrics['Decision Tree'] = {
        'Accuracy': accuracy_score(y_test, y_pred_dt),
        'AUC Score': roc_auc_score(y_test, y_pred_proba_dt),
        'Precision': precision_score(y_test, y_pred_dt),
        'Recall': recall_score(y_test, y_pred_dt),
        'F1 Score': f1_score(y_test, y_pred_dt),
        'MCC Score': matthews_corrcoef(y_test, y_pred_dt)
    }
    models['decision_tree'] = dt_model
    print(f"   ✓ Accuracy: {metrics['Decision Tree']['Accuracy']:.4f}")
    
    # 3. K-Nearest Neighbors
    print("\n3. Training K-Nearest Neighbors...")
    knn_model = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='minkowski', p=2)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    y_pred_proba_knn = knn_model.predict_proba(X_test)[:, 1]
    
    metrics['K-Nearest Neighbors'] = {
        'Accuracy': accuracy_score(y_test, y_pred_knn),
        'AUC Score': roc_auc_score(y_test, y_pred_proba_knn),
        'Precision': precision_score(y_test, y_pred_knn),
        'Recall': recall_score(y_test, y_pred_knn),
        'F1 Score': f1_score(y_test, y_pred_knn),
        'MCC Score': matthews_corrcoef(y_test, y_pred_knn)
    }
    models['knn'] = knn_model
    print(f"   ✓ Accuracy: {metrics['K-Nearest Neighbors']['Accuracy']:.4f}")
    
    return models, metrics

def save_models(models, label_encoders, feature_columns):
    """Save trained models and encoders to disk"""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    # Create models directory if it doesn't exist
    os.makedirs('saved_models', exist_ok=True)
    
    # Save each model
    for model_name, model in models.items():
        filename = f'saved_models/{model_name}.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"✓ Saved: {filename}")
    
    # Save label encoders
    with open('saved_models/label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✓ Saved: saved_models/label_encoders.pkl")
    
    # Save feature columns order
    with open('saved_models/feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_columns, f)
    print(f"✓ Saved: saved_models/feature_columns.pkl")

def main():
    print("="*70)
    print("MUSHROOM CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    df = load_data()
    
    # Preprocess
    df_encoded, label_encoders = preprocess_data(df)
    
    # Split data
    target_col = df.columns[0]
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]
    
    print(f"\nSplitting data (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train models
    models, metrics = train_models(X_train, y_train, X_test, y_test)
    
    # Save models
    save_models(models, label_encoders, list(X.columns))
    
    # Print summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}:")
        for metric, value in model_metrics.items():
            print(f"  {metric:15s}: {value:.4f}")
    
    print("\n" + "="*70)
    print("✓ All models trained and saved successfully!")
    print("="*70)
    print("\nYou can now run the Streamlit app to make predictions:")
    print("  streamlit run app.py")

if __name__ == "__main__":
    main()
