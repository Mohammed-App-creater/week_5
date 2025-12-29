"""
Task 2a: Baseline Model - Logistic Regression

This module implements the baseline Logistic Regression model for fraud detection
on both E-commerce and Credit Card datasets, with proper stratified train-test split
and comprehensive evaluation metrics.

Author: Week 5 - Fraud Detection Project
Date: 2025-12-29
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report
)
import joblib
import warnings

warnings.filterwarnings('ignore')

# Constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def prepare_features_target(df, target_col, drop_cols=None):
    """
    Explicitly separate features (X) and target (y) from dataset.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column
        drop_cols: List of columns to drop (e.g., identifiers)
    
    Returns:
        X: Feature matrix
        y: Target vector
    """
    # Separate target
    y = df[target_col].copy()
    
    # Separate features
    X = df.drop(columns=[target_col])
    
    # Drop identifier columns if specified
    if drop_cols:
        X = X.drop(columns=[col for col in drop_cols if col in X.columns])
    
    # Drop any remaining object (string) columns
    obj_cols = X.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        print(f"  Dropping non-numeric columns: {list(obj_cols)}")
        X = X.drop(columns=obj_cols)
    
    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    
    return X, y


def stratified_train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified 80/20 train-test split.
    
    Args:
        X: Feature matrix
        y: Target vector
        test_size: Proportion of test set (default: 0.2 for 80/20 split)
        random_state: Random seed for reproducibility
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y  # STRATIFIED split to preserve class distribution
    )


def train_baseline_logistic_regression(X_train, y_train):
    """
    Train baseline Logistic Regression model.
    
    Uses class_weight='balanced' to handle class imbalance.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        Trained LogisticRegression model
    """
    model = LogisticRegression(
        class_weight='balanced',  # Handle class imbalance
        max_iter=1000,            # Sufficient iterations for convergence
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model


def evaluate_and_print_metrics(model, X_test, y_test, dataset_name):
    """
    Compute and PRINT all required evaluation metrics.
    
    Metrics computed:
    - ROC-AUC
    - Precision
    - Recall
    - F1-score
    - Confusion Matrix
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        dataset_name: Name of dataset for labeling output
    """
    # Generate predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Compute metrics
    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    # Print results with clear labeling
    print(f"\n{'='*70}")
    print(f"Baseline Logistic Regression – {dataset_name}")
    print(f"{'='*70}")
    print(f"\nEvaluation Metrics:")
    print(f"  ROC-AUC:    {roc_auc:.4f}")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1-score:   {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={cm[0,0]:>6}  FP={cm[0,1]:>6}]")
    print(f"   [FN={cm[1,0]:>6}  TP={cm[1,1]:>6}]]")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"{'='*70}\n")
    
    return {
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm.tolist()
    }


def main():
    """Main execution function for Task 2a - Baseline Model."""
    
    print("\n" + "="*70)
    print("TASK 2a: BASELINE MODEL - LOGISTIC REGRESSION")
    print("="*70)
    
    # ========================================================================
    # E-COMMERCE FRAUD DATA (Fraud_Data.csv)
    # ========================================================================
    print("\n[1/2] Processing E-commerce Fraud Data...")
    
    # Load data
    fraud_df = pd.read_csv('data/processed/Fraud_Data_Processed.csv')
    print(f"  Loaded dataset shape: {fraud_df.shape}")
    
    # 1) Explicitly separate features (X) and target (y)
    #    Target column: "class"
    X_fraud, y_fraud = prepare_features_target(
        fraud_df, 
        target_col='class',
        drop_cols=['user_id']  # Drop identifier column
    )
    print(f"  Features shape: {X_fraud.shape}")
    print(f"  Target distribution: {dict(y_fraud.value_counts())}")
    
    # 2) Perform STRATIFIED 80/20 train-test split
    X_train_f, X_test_f, y_train_f, y_test_f = stratified_train_test_split(
        X_fraud, y_fraud, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train set: {X_train_f.shape}, Test set: {X_test_f.shape}")
    
    # 3) Train Logistic Regression baseline
    print("  Training Logistic Regression (class_weight='balanced')...")
    lr_fraud = train_baseline_logistic_regression(X_train_f, y_train_f)
    
    # 4) Compute and PRINT metrics
    metrics_fraud = evaluate_and_print_metrics(
        lr_fraud, X_test_f, y_test_f, 
        dataset_name="E-commerce Fraud"
    )
    
    # Save model
    model_path_fraud = MODELS_DIR / 'baseline_logistic_regression_fraud.pkl'
    joblib.dump(lr_fraud, model_path_fraud)
    print(f"Model saved to: {model_path_fraud}\n")
    
    # ========================================================================
    # CREDIT CARD FRAUD DATA (creditcard.csv)
    # ========================================================================
    print("\n[2/2] Processing Credit Card Fraud Data...")
    
    # Load data
    credit_df = pd.read_csv('data/processed/creditcard_Processed.csv')
    print(f"  Loaded dataset shape: {credit_df.shape}")
    
    # 1) Explicitly separate features (X) and target (y)
    #    Target column: "Class"
    X_credit, y_credit = prepare_features_target(
        credit_df, 
        target_col='Class'
    )
    print(f"  Features shape: {X_credit.shape}")
    print(f"  Target distribution: {dict(y_credit.value_counts())}")
    
    # 2) Perform STRATIFIED 80/20 train-test split
    X_train_c, X_test_c, y_train_c, y_test_c = stratified_train_test_split(
        X_credit, y_credit, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"  Train set: {X_train_c.shape}, Test set: {X_test_c.shape}")
    
    # 3) Train Logistic Regression baseline
    print("  Training Logistic Regression (class_weight='balanced')...")
    lr_credit = train_baseline_logistic_regression(X_train_c, y_train_c)
    
    # 4) Compute and PRINT metrics
    metrics_credit = evaluate_and_print_metrics(
        lr_credit, X_test_c, y_test_c, 
        dataset_name="Credit Card Fraud"
    )
    
    # Save model
    model_path_credit = MODELS_DIR / 'baseline_logistic_regression_credit.pkl'
    joblib.dump(lr_credit, model_path_credit)
    print(f"Model saved to: {model_path_credit}\n")
    
    print("\n" + "="*70)
    print("TASK 2a COMPLETE - Baseline models trained and evaluated")
    print("="*70 + "\n")
    
    return {
        'fraud': metrics_fraud,
        'credit': metrics_credit
    }


if __name__ == "__main__":
    main()
