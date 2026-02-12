"""
Create train.csv and test.csv from Kaggle mushroom dataset
Splits the dataset 80/20 and saves to CSV files
"""

import pandas as pd
import os
import kagglehub
from sklearn.model_selection import train_test_split

def create_train_test_csv():
    """Download dataset and create train.csv and test.csv"""
    
    print("="*70)
    print("CREATING TRAIN.CSV AND TEST.CSV FILES")
    print("="*70)
    
    # Download dataset from Kaggle
    print("\n1. Downloading dataset from Kaggle...")
    path = kagglehub.dataset_download("uciml/mushroom-classification")
    
    # Find CSV file
    csv_file = None
    for file in os.listdir(path):
        if file.endswith('.csv'):
            csv_file = os.path.join(path, file)
            break
    
    if not csv_file:
        raise FileNotFoundError("CSV file not found in dataset")
    
    # Load dataset
    print("\n2. Loading dataset...")
    df = pd.read_csv(csv_file)
    print(f"   Total samples: {df.shape[0]}")
    print(f"   Total features: {df.shape[1]}")
    print(f"   Columns: {list(df.columns)}")
    
    # Display dataset info
    print("\n3. Dataset preview:")
    print(df.head())
    
    print("\n4. Class distribution:")
    print(df.iloc[:, 0].value_counts())
    
    # Split dataset (80/20)
    print("\n5. Splitting dataset (80% train, 20% test)...")
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify=df.iloc[:, 0]  # Stratify by target column
    )
    
    print(f"   Training set: {train_df.shape[0]} samples")
    print(f"   Test set: {test_df.shape[0]} samples")
    
    # Save to CSV
    print("\n6. Saving files...")
    train_df.to_csv('data/train.csv', index=False)
    print(f"   ✓ Saved: train.csv ({train_df.shape[0]} rows)")

    test_df.to_csv('data/test.csv', index=False)
    print(f"   ✓ Saved: test.csv ({test_df.shape[0]} rows)")
    
    # Also create test_without_labels.csv for prediction-only testing
    test_features = test_df.iloc[:, 1:]  # Remove first column (target)
    test_features.to_csv('data/test_without_labels.csv', index=False)
    print(f"   ✓ Saved: test_without_labels.csv ({test_features.shape[0]} rows, no labels)")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n Files created in current directory:")
    print(f"   1. train.csv              - {train_df.shape[0]} samples (with labels)")
    print(f"   2. test.csv               - {test_df.shape[0]} samples (with labels)")
    print(f"   3. test_without_labels.csv - {test_df.shape[0]} samples (no labels)")
    
    print(f"\n File details:")
    print(f"   train.csv:")
    print(f"     - Use for: Training models")
    print(f"     - Columns: {train_df.shape[1]} (including target)")
    print(f"     - Class distribution:")
    for label, count in train_df.iloc[:, 0].value_counts().items():
        print(f"       {label}: {count} ({count/len(train_df)*100:.1f}%)")
    
    print(f"\n   test.csv:")
    print(f"     - Use for: Evaluation (has labels)")
    print(f"     - Columns: {test_df.shape[1]} (including target)")
    print(f"     - Class distribution:")
    for label, count in test_df.iloc[:, 0].value_counts().items():
        print(f"       {label}: {count} ({count/len(test_df)*100:.1f}%)")
    
    print(f"\n   test_without_labels.csv:")
    print(f"     - Use for: Prediction only (no labels)")
    print(f"     - Columns: {test_features.shape[1]} (features only)")
    
    print("\n" + "="*70)
    print("✓ Files created successfully!")
    print("="*70)
    
    print("\n Next steps:")
    print("   1. Train models:")
    print("      python train_models.py")
    print("\n   2. Test with prediction app:")
    print("      streamlit run app_prediction.py")
    print("      (Upload test.csv or test_without_labels.csv)")

if __name__ == "__main__":
    create_train_test_csv()
