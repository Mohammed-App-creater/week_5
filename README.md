# Fraud Detection Project - Week 5

## Overview

This project implements an end-to-end fraud detection pipeline for two datasets:
1. **E-commerce Fraud Data** (`Fraud_Data.csv`)
2. **Credit Card Fraud Data** (`creditcard.csv`)

The project covers data preprocessing, exploratory data analysis (EDA), feature engineering, and machine learning model development with a focus on handling highly imbalanced datasets.

---

## Repository Structure

```
week_5/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Cleaned, feature-engineered datasets
├── models/                     # Saved trained models
│   ├── baseline_logistic_regression_fraud.pkl
│   ├── baseline_logistic_regression_credit.pkl
│   ├── ensemble_xgboost_fraud.pkl
│   └── ensemble_xgboost_credit.pkl
├── src/                        # Source code
│   ├── task2a_baseline_model.py       # Task 2a: Baseline Logistic Regression
│   ├── task2b_ensemble_model.py       # Task 2b: XGBoost Ensemble with CV
│   ├── data_processing.py             # Data preprocessing pipeline
│   └── eda_analysis.py                # EDA visualization script
├── tests/                      # Unit tests
│   ├── __init__.py
│   └── test_model_training.py
├── notebooks/                  # Jupyter notebooks
│   ├── model_training.ipynb
│   ├── data_processing.ipynb
│   └── eda_analysis.ipynb
├── reports/                    # Reports and figures
│   ├── figures/                # Generated plots
│   ├── task2_report.md         # Task 2 implementation report
│   └── Interim_Report.md       # Task 1 summary
├── README.md
└── requirements.txt
```

---

## Setup and Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages**:
- `pandas`, `numpy` - Data manipulation
- `matplotlib`, `seaborn` - Visualization
- `scikit-learn` - Machine learning
- `xgboost` - Ensemble models
- `imbalanced-learn` - Handling class imbalance
- `joblib` - Model persistence

### 2. Setup Data

Place the following files in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

---

## How to Run

### Task 1: Data Analysis and Preprocessing

#### Step 1: Data Processing
```bash
python src/data_processing.py
```

**Output**:
- `data/processed/Fraud_Data_Processed.csv`
- `data/processed/creditcard_Processed.csv`

**What it does**:
- Cleans data (handles missing values, duplicates)
- Merges geolocation data for fraud dataset
- Engineers features (temporal, behavioral, statistical)
- Handles class imbalance preparation

#### Step 2: Exploratory Data Analysis
```bash
python src/eda_analysis.py
```

**Output**: Visualizations saved to `reports/figures/`

**What it does**:
- Univariate and bivariate analysis
- Class distribution analysis
- Feature correlation heatmaps
- Fraud pattern visualization

---

### Task 2: Model Building & Training

#### Task 2a: Baseline Model (Logistic Regression)

```bash
python src/task2a_baseline_model.py
```

**What it does**:
1. **Explicitly separates** features (X) and target (y) for both datasets
   - E-commerce: target = `"class"`
   - Credit Card: target = `"Class"`
2. **Performs stratified 80/20 train-test split**
3. **Trains Logistic Regression** with `class_weight='balanced'`
4. **Computes and PRINTS** all metrics:
   - ROC-AUC
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
5. **Saves models** to `models/` directory

**Expected Output**:
```
======================================================================
Baseline Logistic Regression – E-commerce Fraud
======================================================================

Evaluation Metrics:
  ROC-AUC:    0.XXXX
  Precision:  0.XXXX
  Recall:     0.XXXX
  F1-score:   0.XXXX

Confusion Matrix:
  [[TN=XXXXXX  FP=XXXXXX]
   [FN=XXXXXX  TP=XXXXXX]]
...
```

#### Task 2b: Ensemble Model (XGBoost with Cross-Validation)

```bash
python src/task2b_ensemble_model.py
```

**What it does**:
1. **Implements XGBoost ensemble** model
2. **Handles class imbalance** via `scale_pos_weight`
3. **Performs Stratified K-Fold Cross-Validation** (k=5)
4. **Reports MEAN ± STD** for:
   - F1-score
   - AUC-PR (Average Precision)
5. **Light hyperparameter tuning**:
   - `n_estimators`: [50, 100, 150]
   - `max_depth`: [4, 6, 8]
6. **Prints clear comparison**: Baseline vs Ensemble
7. **Provides model selection justification**

**Expected Output**:
```
Stratified 5-Fold Cross-Validation Results (E-commerce Fraud):
  F1-score:  0.XXXX ± 0.XXXX
  AUC-PR:    0.XXXX ± 0.XXXX

======================================================================
MODEL COMPARISON: E-commerce Fraud
======================================================================

Metric          Baseline (LR)        Ensemble (XGB)       Winner
----------------------------------------------------------------------
ROC-AUC         0.XXXX               0.XXXX               Ensemble
Precision       0.XXXX               0.XXXX               Ensemble
Recall          0.XXXX               0.XXXX               Ensemble
F1-score        0.XXXX               0.XXXX               Ensemble
...
```

---

## Why These Metrics for Imbalanced Data?

### Standard Accuracy is Misleading
For highly imbalanced datasets (e.g., 1% fraud rate), a model that predicts "no fraud" for everything achieves 99% accuracy but is completely useless.

### Metrics Used

| Metric | Why It Matters |
|--------|----------------|
| **ROC-AUC** | Measures discrimination ability across all thresholds |
| **AUC-PR** | More informative than ROC-AUC for imbalanced data; focuses on minority class |
| **Precision** | Minimizes false positives (reduces customer friction) |
| **Recall** | Minimizes false negatives (reduces financial losses) |
| **F1-score** | Balances precision and recall in a single metric |
| **Confusion Matrix** | Enables cost-benefit analysis based on business requirements |

### Business Impact

- **False Positives (FP)**: Legitimate transactions flagged as fraud
  - Customer friction, support costs, potential churn
  
- **False Negatives (FN)**: Fraudulent transactions missed
  - Direct financial losses, reputation damage

The model selection balances these trade-offs based on business priorities.

---

## Testing

Run unit tests to verify implementation:

```bash
python -m pytest tests/
```

Or run individual test file:

```bash
python tests/test_model_training.py
```

---

## Documentation

- **Task 1 Report**: `Interim_Report.md` - Data analysis and preprocessing
- **Task 2 Report**: `reports/task2_report.md` - Model building and training
- **Figures**: `reports/figures/` - All generated visualizations

---

## Key Features

### Data Processing
- ✅ Geolocation integration via IP address
- ✅ Temporal feature engineering
- ✅ Behavioral feature engineering
- ✅ Statistical aggregations
- ✅ Missing value handling
- ✅ Duplicate removal

### Model Development
- ✅ Stratified train-test split (preserves class distribution)
- ✅ Baseline Logistic Regression with balanced class weights
- ✅ XGBoost ensemble with class imbalance handling
- ✅ Stratified K-Fold Cross-Validation (k=5)
- ✅ Hyperparameter tuning
- ✅ Comprehensive evaluation metrics
- ✅ Model persistence for deployment

### Code Quality
- ✅ Reusable functions with clear docstrings
- ✅ Reproducible results (random_state=42)
- ✅ Professional code structure
- ✅ Unit tests
- ✅ Comprehensive documentation

---

## Reproducibility

All scripts use:
- **Random state**: 42
- **Stratified splitting**: Ensures consistent class distribution
- **Fixed hyperparameters**: Documented in code
- **Saved models**: Can be loaded for inference without retraining

---

## For Reviewers

### Task 2 Implementation Checklist

#### Task 2a - Baseline Model ✅
- [x] Explicit feature/target separation for both datasets
- [x] Stratified 80/20 train-test split
- [x] Logistic Regression with `class_weight='balanced'`
- [x] ROC-AUC, Precision, Recall, F1-score computed and printed
- [x] Confusion Matrix displayed
- [x] Clear output labels for each dataset
- [x] Models saved to `models/` directory

#### Task 2b - Ensemble Model ✅
- [x] XGBoost ensemble implemented
- [x] Class imbalance handled via `scale_pos_weight`
- [x] Stratified K-Fold Cross-Validation (k=5)
- [x] MEAN ± STD reported for F1 and AUC-PR
- [x] Hyperparameter tuning (n_estimators, max_depth)
- [x] Clear baseline vs ensemble comparison
- [x] Model selection justification provided

#### Repository Structure ✅
- [x] `src/` - Reusable training & evaluation functions
- [x] `models/` - Saved trained models
- [x] `tests/` - Basic test stub
- [x] `reports/` - Task 2 report

#### Documentation ✅
- [x] README explains Task 2a and Task 2b
- [x] Instructions on how to run modeling scripts
- [x] Metrics explained with business context
- [x] Clear instructions for reviewers

---

## Contact

For questions or issues, please refer to the documentation in `reports/` or contact the project maintainer.
