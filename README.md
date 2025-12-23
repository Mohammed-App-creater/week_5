# Fraud Detection Project - Week 5

## Structure
- `data/Raw`: Place `Fraud_Data.csv`, `IpAddress_to_Country.csv`, and `creditcard.csv` here.
- `src/data_processing.py`: Main script to clean data, merge geolocation, engineer features, and save processed data to `data/processed/`.
- `src/eda_analysis.py`: Generates EDA plots and saves them to `reports/figures/`.
- `Interim_Report.md`: Summary of analysis.

## How to Run

1. **Install Dependencies** (Ensure you have `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `imbalanced-learn` installed).
2. **Setup Data**:
   Ensure the following files are in `data/raw/`:
   - `Fraud_Data.csv`
   - `IpAddress_to_Country.csv`
   - `creditcard.csv`
3. **Run Processing**:
   ```bash
   python src/data_processing.py
   ```
   This will create `Fraud_Data_Processed.csv` and `creditcard_Processed.csv` in `data/processed/`.
4. **Run EDA**:
   ```bash
   python src/eda_analysis.py
   ```
   Check `reports/figures/` for the generated plots.
