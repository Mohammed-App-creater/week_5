# Interim-1 Report: Fraud Detection for E-commerce and Banking

**Researcher:** Antigravity (Senior Data Scientist)  
**Project:** Fraud Detection Pipeline Development  
**Date:** December 23, 2025  

---

## 1. Introduction
The objective of this project is to build a robust fraud detection system for both e-commerce transactions and credit card operations. Fraudulent activities pose a significant financial risk and undermine consumer trust. Task 1 focused on building a foundational data pipeline, including data cleaning, exploratory data analysis (EDA), and feature engineering, to prepare high-quality datasets for predictive modeling.

## 2. Data Overview
The project utilizes two distinct datasets to capture different fraud patterns:

### E-commerce Fraud Dataset (`Fraud_Data.csv`)
- **Size:** 151,112 records
- **Target Variable:** `class` (1 = Fraud, 0 = Legitimate)
- **Class Imbalance:** 
  - Legitimate: 136,961 (90.64%)
  - Fraud: 14,151 (9.36%)
- **Context:** Individual user transactions with metadata such as IP address, browser, and device ID.

### Credit Card Transaction Dataset (`creditcard_Processed.csv`)
- **Size:** 283,726 records (after cleaning)
- **Target Variable:** `Class` (1 = Fraud, 0 = Legitimate)
- **Class Imbalance:** 
  - Legitimate: 283,253 (99.83%)
  - Fraud: 473 (0.17%)
- **Context:** Anonymized transaction features (V1-V28) with Amount and Time.

## 3. Data Cleaning
To ensure data quality, the following preprocessing steps were implemented:

- **Missing Values:** Imputed using the median for numerical columns and the mode for categorical columns.
- **Duplicates:** Removed 1,081 duplicate rows from the credit card dataset to prevent model bias.
- **Data Types:** Converted `signup_time` and `purchase_time` to datetime objects for temporal analysis.
- **Geolocation Mapping:** Merged IP addresses from `Fraud_Data.csv` with `IpAddress_to_Country.csv` using a range-based lookup to identify the originating country of each transaction.

## 4. Exploratory Data Analysis (EDA)
EDA revealed critical patterns that differentiate fraudulent behavior from legitimate usage.

### 4.1 E-commerce Fraud Analysis

#### Univariate Analysis
We examined the distribution of individual features to understand the profile of transactions.

![Fraud Univariate Age](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/Fraud_univariate_age.png)
*Figure 1: Age distribution in the Fraud dataset.*

![Fraud Univariate Purchase Value](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/Fraud_univariate_purchase_value.png)
*Figure 2: Purchase value distribution in the Fraud dataset.*

![Fraud Univariate Browser](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/Fraud_univariate_browser.png)
*Figure 3: Browser distribution.*

![Fraud Univariate Sex](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/Fraud_univariate_sex.png)
*Figure 4: Gender distribution.*

![Fraud Univariate Source](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/Fraud_univariate_source.png)
*Figure 5: Marketing source distribution.*

#### Bivariate Analysis
Analyzing the relationship between features and the target `class` highlights discriminative patterns.

![Fraud Bivariate Age vs Class](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/Fraud_bivariate_age_vs_class.png)
*Figure 6: Age vs Fraudulent transactions.*

![Fraud Bivariate Purchase Value vs Class](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/Fraud_bivariate_purchase_value_vs_class.png)
*Figure 7: Purchase Value vs Fraudulent transactions.*

### 4.2 Credit Card Transaction Analysis

#### Univariate Analysis
Distribution of transaction amounts and time for the credit card dataset.

![Credit Card Univariate Amount](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/CreditCard_univariate_Amount.png)
*Figure 8: Distribution of transaction amounts.*

![Credit Card Univariate Time](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/CreditCard_univariate_Time.png)
*Figure 9: Distribution of transaction time.*

#### Bivariate Analysis
Comparison of transaction patterns between legitimate and fraudulent classes.

![Credit Card Bivariate Amount vs Class](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/CreditCard_bivariate_Amount_vs_Class.png)
*Figure 10: Amount vs Class.*

![Credit Card Bivariate Time vs Class](file:///c:/Users/yoga/code/10_Academy/week_5/reports/figures/CreditCard_bivariate_Time_vs_Class.png)
*Figure 11: Time vs Class.*

### 4.3 Geolocation & Fraud Patterns
Mapping IP addresses to countries allowed us to visualize fraud rates by region. This feature is crucial as certain jurisdictions may exhibit higher frequencies of fraudulent activity.

## 5. Feature Engineering
We engineered several features to capture behavioral and temporal signals of fraud:

- **`hour_of_day` & `day_of_week`:** Captures the timing of transactions. Fraudulent transactions often occur at unusual hours.
- **`time_since_signup`:** The duration (in seconds) between user registration and their first purchase. Rapid transactions after signup are a high-risk indicator.
- **`transaction_count`:** Cumulative count of transactions per `user_id` or `device_id` to detect high-frequency automated attacks.
- **`country`:** Categorical feature derived from IP ranges to capture geographic risk.

## 6. Class Imbalance Handling
The extreme imbalance in the Credit Card dataset (0.17% fraud) necessitates specialized handling. We have prepared the pipeline to use **SMOTE (Synthetic Minority Over-sampling Technique)** during the training phase.

- **Current State:** Highly imbalanced datasets.
- **Proposed Handling:** Re-balancing the minority class (Fraud) in the training set to ensure the model learns to identify fraud signatures without being overwhelmed by legitimate examples.

## 7. Summary & Key Insights
1. **Time-to-Transaction is Key:** Fraudulent users often make purchases almost immediately after signing up, making `time_since_signup` a powerful predictor.
2. **Extreme Imbalance in Banking:** Credit card fraud is rare (0.17%), requiring robust evaluation metrics like Precision-Recall AUC rather than simple Accuracy.
3. **Consistency Across Browsers:** Fraud rates were relatively uniform across major browsers (Chrome, Safari, IE), suggesting that device/IP-level features are more discriminative than software-level ones.
4. **Volume Signals Fraud:** High `transaction_count` per device in a short window is a strong indicator of automated fraud bots.

---

## References
- Kaggle Dataset: [Fraud Detection](https://www.kaggle.com/)
- Scikit-learn Documentation: [Preprocessing & Scaling](https://scikit-learn.org/stable/modules/preprocessing.html)
- Imbalanced-learn: [SMOTE Documentation](https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
- Python Logging: [Logging Library Guide](https://docs.python.org/3/library/logging.html)

---

## Appendix: Preprocessing Logs
Preprocessing was executed with Python's `logging` module to ensure auditability. Key logs included:
- `Dropped 1081 duplicate rows in creditcard dataset.`
- `Converted signup_time to datetime.`
- `Integrated geolocation data using merge_asof.`
