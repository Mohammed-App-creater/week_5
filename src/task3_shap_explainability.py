"""
Task 3: Model Explainability with SHAP

This script performs comprehensive SHAP-based explainability analysis on trained XGBoost
fraud detection models. It generates:
1. Baseline feature importance from XGBoost
2. Global SHAP explanations (summary plots, bar plots)
3. Local SHAP explanations (force plots for TP, FP, FN cases)
4. Analysis results saved for report generation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import joblib
import shap
import json
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
FIGURES_DIR = Path('reports/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('ggplot')
sns.set_palette("husl")


def load_model_and_data(dataset_name):
    """
    Load trained model and processed data for analysis.
    
    Args:
        dataset_name: 'fraud' or 'credit'
    
    Returns:
        model, X_test, y_test, feature_names
    """
    if dataset_name == 'fraud':
        model_path = 'models/ensemble_xgboost_fraud.pkl'
        data_path = 'data/processed/Fraud_Data_Processed.csv'
        target_col = 'class'
        drop_cols = ['user_id']
    else:
        model_path = 'models/ensemble_xgboost_credit.pkl'
        data_path = 'data/processed/creditcard_Processed.csv'
        target_col = 'Class'
        drop_cols = []
    
    # Load model
    print(f"\n{'='*60}")
    print(f"Loading {dataset_name.upper()} dataset model and data...")
    print(f"{'='*60}")
    
    try:
        model = joblib.load(model_path)
        print(f"✓ Model loaded from {model_path}")
    except FileNotFoundError:
        print(f"✗ Model not found at {model_path}")
        return None, None, None, None
    
    # Load data
    df = pd.read_csv(data_path)
    print(f"✓ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    if drop_cols:
        X = X.drop(columns=[col for col in drop_cols if col in X.columns])
    
    # Drop object columns
    obj_cols = X.select_dtypes(include=['object']).columns
    if len(obj_cols) > 0:
        X = X.drop(columns=obj_cols)
    
    # Convert boolean to int
    bool_cols = X.select_dtypes(include=['bool']).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)
    
    y = df[target_col]
    
    # Split data (same as training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    
    print(f"✓ Test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"  - Fraud cases: {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"  - Legitimate cases: {(~y_test.astype(bool)).sum()} ({(1-y_test.mean())*100:.2f}%)")
    
    return model, X_test, y_test, X_test.columns.tolist()


def plot_baseline_feature_importance(model, feature_names, dataset_name, top_n=10):
    """
    Extract and visualize XGBoost built-in feature importance.
    """
    print(f"\n--- Baseline Feature Importance ({dataset_name.upper()}) ---")
    
    # Get feature importance (gain-based)
    importance = model.get_booster().get_score(importance_type='gain')
    
    # Map to feature names
    importance_df = pd.DataFrame([
        {
            'feature': (
                feature_names[int(k[1:])] if k.startswith('f') and k[1:].isdigit() else k
            ),
            'importance': v
        }
        for k, v in importance.items()
    ]).sort_values('importance', ascending=False).head(top_n)

    
    print(f"Top {top_n} features by XGBoost gain:")
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.2f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'].values, color='steelblue')
    plt.yticks(range(len(importance_df)), importance_df['feature'].values)
    plt.xlabel('Feature Importance (Gain)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'Top {top_n} Features - XGBoost Built-in Importance ({dataset_name.upper()})', fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    save_path = FIGURES_DIR / f'{dataset_name}_xgb_feature_importance.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved to {save_path}")
    
    return importance_df


def compute_shap_values(model, X_test, dataset_name, sample_size=1000):
    print(f"\n--- Computing SHAP Values ({dataset_name.upper()}) ---")

    # Sample for speed
    if len(X_test) > sample_size:
        print(f"Sampling {sample_size} instances for SHAP computation...")
        X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
    else:
        X_sample = X_test

    print("Initializing SHAP explainer (probability output)...")

    # IMPORTANT: explain fraud probability (class 1)
    explainer = shap.Explainer(
        model.predict_proba,
        X_sample
    )

    print("Computing SHAP values...")
    shap_values = explainer(X_sample)

    # Take SHAP values for positive (fraud) class
    shap_values = shap_values.values[:, :, 1]

    print(f"✓ SHAP values computed: shape {shap_values.shape}")

    return explainer, shap_values, X_sample


def plot_shap_summary(shap_values, X_sample, dataset_name, top_n=15):
    """
    Generate SHAP summary plots (beeswarm and bar).
    """
    print(f"\n--- SHAP Global Explainability ({dataset_name.upper()}) ---")
    
    # Beeswarm plot (summary plot)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False, max_display=top_n)
    plt.title(f'SHAP Summary Plot - Feature Impact Distribution ({dataset_name.upper()})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = FIGURES_DIR / f'{dataset_name}_shap_summary_beeswarm.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Beeswarm plot saved to {save_path}")
    
    # Bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=top_n)
    plt.title(f'SHAP Feature Importance - Mean Absolute Impact ({dataset_name.upper()})', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    save_path = FIGURES_DIR / f'{dataset_name}_shap_summary_bar.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Bar plot saved to {save_path}")
    
    # Get top features by mean absolute SHAP
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': mean_abs_shap
    }).sort_values('mean_abs_shap', ascending=False).head(5)
    
    print(f"\nTop 5 features by mean absolute SHAP value:")
    for idx, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    return top_features


def analyze_local_predictions(model, X_test, y_test, shap_values, X_sample, dataset_name):
    """
    Identify and analyze specific prediction cases (TP, FP, FN).
    """
    print(f"\n--- Local Prediction Analysis ({dataset_name.upper()}) ---")
    
    # Get predictions on full test set
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Confusion matrix components
    cm = confusion_matrix(y_test, y_pred)
    
    # Find indices for each case
    tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    
    print(f"Confusion Matrix:")
    print(f"  True Positives: {len(tp_indices)}")
    print(f"  False Positives: {len(fp_indices)}")
    print(f"  False Negatives: {len(fn_indices)}")
    print(f"  True Negatives: {cm[0, 0]}")
    
    cases = {}
    
    # Select one example from each category
    if len(tp_indices) > 0:
        tp_idx = tp_indices[0]
        cases['TP'] = {
            'index': int(tp_idx),
            'actual': int(y_test.iloc[tp_idx]),
            'predicted': int(y_pred[tp_idx]),
            'probability': float(y_prob[tp_idx]),
            'features': X_test.iloc[tp_idx].to_dict()
        }
        print(f"\n✓ True Positive example: index {tp_idx}, prob={y_prob[tp_idx]:.4f}")
    
    if len(fp_indices) > 0:
        fp_idx = fp_indices[0]
        cases['FP'] = {
            'index': int(fp_idx),
            'actual': int(y_test.iloc[fp_idx]),
            'predicted': int(y_pred[fp_idx]),
            'probability': float(y_prob[fp_idx]),
            'features': X_test.iloc[fp_idx].to_dict()
        }
        print(f"✓ False Positive example: index {fp_idx}, prob={y_prob[fp_idx]:.4f}")
    
    if len(fn_indices) > 0:
        fn_idx = fn_indices[0]
        cases['FN'] = {
            'index': int(fn_idx),
            'actual': int(y_test.iloc[fn_idx]),
            'predicted': int(y_pred[fn_idx]),
            'probability': float(y_prob[fn_idx]),
            'features': X_test.iloc[fn_idx].to_dict()
        }
        print(f"✓ False Negative example: index {fn_idx}, prob={y_prob[fn_idx]:.4f}")
    
    # Generate force plots for cases that exist in X_sample
    for case_type, case_data in cases.items():
        try:
            # Find this instance in X_sample
            sample_idx = X_test.index.get_loc(case_data['index'])
            if sample_idx < len(X_sample):
                plot_force_plot(shap_values, X_sample, sample_idx, case_type, dataset_name)
        except:
            print(f"  Note: {case_type} case not in SHAP sample, skipping force plot")
    
    return cases


def plot_force_plot(shap_values, X_sample, idx, case_type, dataset_name):
    """
    Generate and save SHAP force plot for a specific prediction.
    """
    try:
        # Create force plot
        shap.initjs()
        
        # Get SHAP values for this instance
        instance_shap = shap_values[idx]
        instance_features = X_sample.iloc[idx]
        
        # Get top contributing features
        feature_contributions = pd.DataFrame({
            'feature': X_sample.columns,
            'shap_value': instance_shap,
            'feature_value': instance_features.values
        })
        feature_contributions['abs_shap'] = np.abs(feature_contributions['shap_value'])
        top_contrib = feature_contributions.nlargest(5, 'abs_shap')
        
        print(f"\n  {case_type} - Top 5 feature contributions:")
        for _, row in top_contrib.iterrows():
            direction = "→ FRAUD" if row['shap_value'] > 0 else "→ LEGIT"
            print(f"    {row['feature']} = {row['feature_value']:.3f}: SHAP = {row['shap_value']:.4f} {direction}")
        
        # Save force plot as matplotlib figure
        plt.figure(figsize=(14, 3))
        
        # Sort by SHAP value for visualization
        sorted_indices = np.argsort(np.abs(instance_shap))[::-1][:10]
        sorted_features = [X_sample.columns[i] for i in sorted_indices]
        sorted_shap = instance_shap[sorted_indices]
        
        colors = ['red' if v > 0 else 'blue' for v in sorted_shap]
        plt.barh(range(len(sorted_shap)), sorted_shap, color=colors, alpha=0.7)
        plt.yticks(range(len(sorted_shap)), sorted_features)
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=11)
        plt.ylabel('Feature', fontsize=11)
        plt.title(f'{case_type} Case - Feature Contributions ({dataset_name.upper()})', 
                  fontsize=13, fontweight='bold')
        plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        save_path = FIGURES_DIR / f'{dataset_name}_shap_force_{case_type.lower()}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Force plot saved to {save_path}")
        
    except Exception as e:
        print(f"  ✗ Could not generate force plot for {case_type}: {e}")


def main():
    """
    Main execution function.
    """
    print("\n" + "="*60)
    print("TASK 3: MODEL EXPLAINABILITY WITH SHAP")
    print("="*60)
    
    results = {}
    
    # Analyze both datasets
    for dataset_name in ['fraud', 'credit']:
        print(f"\n\n{'#'*60}")
        print(f"# ANALYZING {dataset_name.upper()} DATASET")
        print(f"{'#'*60}")
        
        # Load model and data
        model, X_test, y_test, feature_names = load_model_and_data(dataset_name)
        
        if model is None:
            print(f"Skipping {dataset_name} dataset (model not found)")
            continue
        
        # Baseline feature importance
        baseline_importance = plot_baseline_feature_importance(model, feature_names, dataset_name)
        
        # Compute SHAP values
        explainer, shap_values, X_sample = compute_shap_values(model, X_test, dataset_name)
        
        # Global SHAP explanations
        top_shap_features = plot_shap_summary(shap_values, X_sample, dataset_name)
        
        # Local SHAP explanations
        cases = analyze_local_predictions(model, X_test, y_test, shap_values, X_sample, dataset_name)
        
        # Store results
        results[dataset_name] = {
            'baseline_importance': baseline_importance.to_dict('records'),
            'top_shap_features': top_shap_features.to_dict('records'),
            'local_cases': cases
        }
    
    # Save results
    output_path = 'reports/shap_analysis_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'='*60}")
    print(f"✓ SHAP analysis complete!")
    print(f"✓ Results saved to {output_path}")
    print(f"✓ Figures saved to {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
