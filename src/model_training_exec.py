import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, confusion_matrix, 
    classification_report, average_precision_score, PrecisionRecallDisplay
)
from xgboost import XGBClassifier
import joblib
import warnings
import json

warnings.filterwarnings('ignore')

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('ggplot')
RANDOM_STATE = 42

FIGURES_DIR = Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

def prepare_data(df, target_col, drop_cols=None):
    X = df.drop(columns=[target_col])
    if drop_cols:
        X = X.drop(columns=[col for col in drop_cols if col in X.columns])
    
    # Drop any remaining object (string) columns
    obj_cols = X.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        X = X.drop(columns=obj_cols)
    
    # Explicitly convert boolean columns to int to avoid issues with sklearn/xgboost
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test, model_name, dataset_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    f1 = f1_score(y_test, y_pred)
    auc_pr = average_precision_score(y_test, y_prob)
    
    print(f"\n--- {model_name} ({dataset_name}) Evaluation ---")
    print(f"F1-Score: {f1:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")
    
    # Save Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name} ({dataset_name})')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(FIGURES_DIR / f"{dataset_name.lower()}_{model_name.lower().replace(' ', '_')}_cm.png")
    plt.close()
    
    return {
        'model_name': model_name,
        'dataset': dataset_name,
        'f1': float(f1),
        'auc_pr': float(auc_pr),
        'y_prob': y_prob.tolist(),
        'y_test': y_test.tolist()
    }

def run_cv(model, X, y, name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
    auc_pr_scores = cross_val_score(model, X, y, cv=skf, scoring='average_precision')
    
    return {
        'f1_mean': float(f1_scores.mean()),
        'f1_std': float(f1_scores.std()),
        'auc_pr_mean': float(auc_pr_scores.mean()),
        'auc_pr_std': float(auc_pr_scores.std())
    }

def main():
    # Load data
    fraud_df = pd.read_csv('data/processed/Fraud_Data_Processed.csv')
    credit_df = pd.read_csv('data/processed/creditcard_Processed.csv')
    
    # Prepare data
    X_train_f, X_test_f, y_train_f, y_test_f = prepare_data(fraud_df, 'class', drop_cols=['user_id'])
    X_train_c, X_test_c, y_train_c, y_test_c = prepare_data(credit_df, 'Class')
    
    results = []
    
    # Logistic Regression
    lr_f = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr_f.fit(X_train_f, y_train_f)
    results.append(evaluate_model(lr_f, X_test_f, y_test_f, 'Logistic Regression', 'Fraud'))
    
    lr_c = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr_c.fit(X_train_c, y_train_c)
    results.append(evaluate_model(lr_c, X_test_c, y_test_c, 'Logistic Regression', 'Credit'))
    
    # XGBoost
    scale_f = len(y_train_f[y_train_f == 0]) / len(y_train_f[y_train_f == 1])
    xgb_f = XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=scale_f, random_state=RANDOM_STATE, eval_metric='aucpr')
    xgb_f.fit(X_train_f, y_train_f)
    results.append(evaluate_model(xgb_f, X_test_f, y_test_f, 'XGBoost', 'Fraud'))
    
    scale_c = len(y_train_c[y_train_c == 0]) / len(y_train_c[y_train_c == 1])
    xgb_c = XGBClassifier(n_estimators=100, max_depth=6, scale_pos_weight=scale_c, random_state=RANDOM_STATE, eval_metric='aucpr')
    xgb_c.fit(X_train_c, y_train_c)
    results.append(evaluate_model(xgb_c, X_test_c, y_test_c, 'XGBoost', 'Credit'))
    
    # CV
    cv_results = {
        'LogReg_Fraud': run_cv(lr_f, X_train_f, y_train_f, 'LogReg Fraud'),
        'XGBoost_Fraud': run_cv(xgb_f, X_train_f, y_train_f, 'XGBoost Fraud'),
        'LogReg_Credit': run_cv(lr_c, X_train_c, y_train_c, 'LogReg Credit'),
        'XGBoost_Credit': run_cv(xgb_c, X_train_c, y_train_c, 'XGBoost Credit')
    }
    
    # Plot PR Curves
    for dataset in ['Fraud', 'Credit']:
        plt.figure(figsize=(10, 6))
        for res in results:
            if res['dataset'] == dataset:
                label = f"{res['model_name']} (AUC-PR={res['auc_pr']:.3f})"
                PrecisionRecallDisplay.from_predictions(res['y_test'], res['y_prob'], name=label, ax=plt.gca())
        
        y_test_mean = np.mean(y_test_f if dataset == 'Fraud' else y_test_c)
        plt.axhline(y=y_test_mean, color='r', linestyle='--', label='No Skill')
        plt.title(f'Precision-Recall Curve Comparison - {dataset} Data')
        plt.legend()
        plt.savefig(FIGURES_DIR / f"{dataset.lower()}_pr_comparison.png")
        plt.close()

    # Save summary
    summary = {
        'test_results': [{k: v for k, v in r.items() if k not in ['y_prob', 'y_test']} for r in results],
        'cv_results': cv_results
    }
    with open('reports/modeling_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    print("\nModeling summary saved to reports/modeling_summary.json")

if __name__ == "__main__":
    main()
