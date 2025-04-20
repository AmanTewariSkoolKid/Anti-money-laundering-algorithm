# Anti-Money Laundering (AML) Algorithm

A machine learning-based system for detecting suspicious financial transactions that may indicate money laundering activities. This project includes data generation, preprocessing, model training, evaluation, and visualization components for AML compliance and risk management.

## Overview

This project implements a complete AML detection system using machine learning techniques. The system analyzes customer profiles, transaction patterns, and country risk factors to identify potentially suspicious activities. Features include:

- Synthetic data generation for testing and development
- Customer profile risk scoring
- Transaction pattern analysis  
- Machine learning-based anomaly detection
- Visualization of results and model performance
- Exportable reports of suspicious activities

## Project Structure

```
├── aml_model_random_forest.pkl    # Trained model file
├── country_risk_ratings.csv       # Country risk assessment data
├── customer_profiles.csv          # Customer demographic data
├── gen_all.py                     # Data generation script
├── generate_customer_profiles.py  # Customer data generator
├── generate_labeled_cases.py      # Label generation for training data
├── generate_transactions.py       # Transaction data generator
├── labeled_cases.csv              # Labeled data for model training
├── main.py                        # Main AML detection script
├── transactions.csv               # Transaction dataset
└── results/                       # Output directory for results
    ├── aml_detection_results_*.txt         # Detection result reports
    ├── confusion_matrix_*.png              # Model evaluation metrics
    ├── feature_importance_*.png            # Feature importance chart
    ├── pr_curve_*.png                      # Precision-recall curve
    ├── roc_curve_*.png                     # ROC curve visualization
    ├── suspicious_customers_*.csv          # Flagged customers report
    └── suspicious_transactions_*.csv       # Flagged transactions report
```

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap
- joblib

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Anti-money-laundering-algorithm.git
cd Anti-money-laundering-algorithm
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap joblib
```

## Usage

### Generating Test Data

If you need to generate synthetic data for testing:

```bash
python gen_all.py
```

This will create the following files:
- `customer_profiles.csv` - Synthetic customer data
- `transactions.csv` - Synthetic transaction data
- `country_risk_ratings.csv` - Country risk assessment data
- `labeled_cases.csv` - Labeled cases for model training

You can modify the number of customers and transactions in the script.

### Running the AML Detection System

```bash
python main.py
```

This will:
1. Load and validate the data
2. Train a Random Forest classifier on the labeled data
3. Evaluate the model performance
4. Generate visualizations and reports
5. Export lists of suspicious customers and transactions

### Output and Results

The system generates several outputs in the `results/` directory:
- Text report with model performance metrics
- Visualizations of model performance (ROC curve, PR curve, etc.)
- List of suspicious customers identified
- Detailed suspicious transactions with risk scores

## How It Works

The AML detection system works through the following steps:

1. **Data Loading & Preprocessing**:
   - Loads customer profiles, transaction data, and country risk ratings
   - Validates data consistency and fixes issues automatically
   - Prepares features for the ML model

2. **Feature Engineering**:
   - Calculates transaction statistics per customer
   - Incorporates country risk factors
   - Analyzes transaction patterns and behaviors 

3. **Model Training**:
   - Trains a Random Forest model on labeled suspicious cases
   - Uses grid search for hyperparameter optimization
   - Evaluates model performance with cross-validation

4. **Risk Detection**:
   - Applies the model to all customers and transactions
   - Ranks customers and transactions by risk score
   - Flags suspicious activities for investigation

5. **Reporting**:
   - Generates performance metrics reports
   - Creates visualizations of model effectiveness
   - Exports detailed lists of suspicious activities

## Model Performance

The system evaluates model performance using:
- Precision, recall, and F1 score
- ROC curve and AUC
- Precision-recall curve
- Confusion matrix
- Feature importance analysis

## License

This project is licensed under the terms of the license included in the repository.

## Contributing

Contributions to improve the algorithm or extend its capabilities are welcome. Please feel free to submit a pull request.

## Disclaimer

This system is designed for educational and illustrative purposes. Deployment in a production environment would require additional features, security measures, and compliance considerations specific to your regulatory environment.
