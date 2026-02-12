# Mushroom Classification - Multiple ML Models Comparison

## GitHub Repository
[https://github.com/kanavnarula/Multiple-Classification-Models]

## Streamlit App Link
[Add your deployed Streamlit app link here]

## Problem Statement
To classify mushrooms as either edible or poisonous based on their physical characteristics using multiple machine learning algorithms and compare their performance.

## Dataset Description
- **Dataset Name**: Mushroom Classification Dataset
- **Source**: UCI Machine Learning Repository
- **Total Samples**: 8,124 instances
- **Features**: 22 categorical features
- **Classes**: 2 (Edible and Poisonous)
- **Training Set**: 80% (6,499 samples)
- **Test Set**: 20% (1,625 samples)

## Model Performance Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|-----|-----------|--------|----------|-----|
| **Logistic Regression** | 0.9557 | 0.9821 | 0.9599 | 0.9476 | 0.9537 | 0.9113 |
| **Decision Tree** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **kNN** | 0.9969 | 1.0000 | 0.9949 | 0.9987 | 0.9968 | 0.9938 |
| **Naive Bayes** | 0.9286 | 0.9506 | 0.9195 | 0.9336 | 0.9265 | 0.8572 |
| **Random Forest (Ensemble)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **XGBoost (Ensemble)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Model Performance Observations

| ML Model Name | Observation about Model Performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Achieves strong baseline performance with 95.57% accuracy. Fast training and prediction. Despite being a linear model, delivers robust results with good precision (0.96) and recall (0.95). Best for scenarios requiring interpretability and speed. |
| **Decision Tree** | Perfect performance with 100% accuracy across all metrics. The model successfully captures all decision rules from the mushroom features. Fully interpretable with clear decision paths. May be prone to overfitting on new unseen data despite perfect test performance. |
| **kNN** | Excellent performance with 99.69% accuracy. Very close to perfect classification with precision of 0.99 and recall of 0.99. No training phase makes it flexible for updates. Slightly slower prediction time due to distance calculations. |
| **Naive Bayes** | Lowest performance among all models with 92.86% accuracy. Despite the "naive" independence assumption, still achieves competitive results. Extremely fast training and prediction. The Gaussian variant works reasonably well but independence assumption limits performance on this dataset. |
| **Random Forest (Ensemble)** | Perfect performance with 100% accuracy across all metrics. The ensemble of 100 decision trees provides superior generalization. Feature importance reveals key mushroom characteristics. Robust and reliable for production deployment. |
| **XGBoost (Ensemble)** | Perfect performance with 100% accuracy across all metrics. State-of-the-art gradient boosting achieves flawless classification. Built-in regularization ensures generalization. Recommended for production deployment due to best-in-class performance and robustness. |

## Screenshots
![Mushroom Classification App](images/Screenshot%202026-02-12%20at%207.29.26%20AM.png)
