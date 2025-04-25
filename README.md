# Anti-Money Laundering (AML) Algorithm

A machine learning-based system for detecting suspicious financial transactions that may indicate money laundering activities. This project includes data generation, preprocessing, model training, evaluation, and visualization components for AML compliance and risk management.

## Overview

This project implements a complete AML detection system using machine learning techniques. The system analyzes customer profiles, transaction patterns, and country risk factors to identify potentially suspicious activities. Features include:

- Synthetic data generation for testing and development
- Customer profile risk scoring
- Transaction pattern analysis
- Machine learning-based anomaly detection
- Data validation and automatic inconsistency fixing
- Visualization of results and model performance
- Exportable reports of suspicious activities
- Time-stamped results for audit trails

## Suspicious Transaction Detection Criteria

The AML algorithm uses the following factors to determine if a transaction or customer activity is suspicious:

### Customer Risk Factors:
- **Politically Exposed Person (PEP) Status**: Customers flagged as politically exposed persons receive heightened scrutiny
- **Sanctions Match**: Customers who match against sanctions lists are automatically considered high risk
- **Address Change Frequency**: Frequent changes of address may indicate attempts to avoid detection
- **Age and Occupation**: Certain combinations of age and occupation with unusual transaction patterns are flagged
- **Nationality**: Transactions involving high-risk jurisdictions receive additional scrutiny

### Transaction Risk Factors:
- **Large Transaction Amounts**: Unusually large transactions relative to customer history
- **Cash Transactions**: High frequency or volume of cash transactions
- **Cross-Border Activity**: Transactions involving multiple jurisdictions, especially high-risk countries
- **Transaction Patterns**: Unusual patterns like rapid deposits followed by withdrawals (layering)
- **Country Risk Rating**: Transactions involving countries with high risk ratings for financial crimes
- **Transaction Type Distribution**: Unusual distribution of transaction types compared to normal patterns

### Aggregated Customer Behavior:
- **Transaction Velocity**: Sudden increase in transaction frequency
- **Transaction Volume**: Significant changes in total transaction amounts
- **Cash to Non-Cash Ratio**: High percentage of cash transactions relative to electronic transactions
- **Transaction Amount Variability**: Highly variable transaction amounts that don't follow regular patterns

The model calculates a composite risk score from these factors, with indicators weighted by their predictive power as determined during model training. The Random Forest classifier makes the final determination by analyzing these features collectively rather than using simple threshold rules.

## Project Structure

```
├── aml_model_random_forest.pkl    # Trained model file
├── gen_all.py                     # Data generation script
├── generate_customer_profiles.py  # Customer data generator
├── generate_labeled_cases.py      # Label generation for training data
├── generate_transactions.py       # Transaction data generator
├── main.py                        # Main AML detection script
├── README.md                      # Project documentation
├── data_csv/                      # Directory for data files
│   ├── country_risk_ratings.csv   # Country risk assessment data
│   ├── customer_profiles.csv      # Customer demographic data
│   ├── labeled_cases.csv          # Labeled data for model training
│   └── transactions.csv           # Transaction dataset
└── results/                       # Output directory for results
    └── results_YYYYMMDD_HHMMSS/   # Timestamped results directory
        ├── aml_detection_results_*.txt         # Detection result reports
        ├── confusion_matrix.png                # Model evaluation metrics
        ├── feature_importance_*.png            # Feature importance chart
        ├── pr_curve.png                        # Precision-recall curve
        ├── roc_curve.png                       # ROC curve visualization
        ├── suspicious_customers_*.csv          # Flagged customers report
        ├── suspicious_transactions_*.csv       # Flagged transactions report
        ├── country_risk_ratings.csv            # Copy of input data
        ├── customer_profiles.csv               # Copy of input data
        ├── transactions.csv                    # Copy of input data
        └── labeled_cases.csv                   # Copy of input data
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
git clone https://github.com/AmanTewariSkoolKid//Anti-money-laundering-algorithm.git
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

This will create the following files in the `data_csv` directory:
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
2. Automatically fix data inconsistencies if possible
3. Train a Random Forest classifier on the labeled data
4. Evaluate the model performance
5. Generate visualizations and reports
6. Export lists of suspicious customers and transactions
7. Create a timestamped results folder with all outputs and data copies for audit purposes

### Output and Results

The system generates several outputs in the `results/results_YYYYMMDD_HHMMSS/` directory:
- Text report with model performance metrics
- Visualizations of model performance (ROC curve, PR curve, etc.)
- List of suspicious customers identified and ranked by risk score
- Detailed suspicious transactions with risk scores
- Copies of input data files for reproducibility and audit trails

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
   - Calculates transaction velocity and variability metrics

3. **Model Training**:
   - Trains a Random Forest model on labeled suspicious cases
   - Uses grid search for hyperparameter optimization
   - Evaluates model performance with cross-validation
   - Supports multiple model types (Random Forest, Gradient Boosting, Logistic Regression)

4. **Risk Detection**:
   - Applies the model to all customers and transactions
   - Ranks customers and transactions by risk score
   - Flags suspicious activities for investigation
   - Calculates transaction-specific risk scores

5. **Reporting**:
   - Generates performance metrics reports
   - Creates visualizations of model effectiveness
   - Exports detailed lists of suspicious activities
   - Provides feature importance analysis

## Model Customization

The AML detection system supports different model types:

1. **Random Forest (default)**: Best for capturing complex patterns and interactions between features
```python
# To use Random Forest model
aml_model.train_model(model_type='random_forest')
```

2. **Gradient Boosting**: May provide higher accuracy in some cases
```python
# To use Gradient Boosting model
aml_model.train_model(model_type='gradient_boosting')
```

3. **Logistic Regression**: Simpler model with more interpretable coefficients
```python
# To use Logistic Regression model
aml_model.train_model(model_type='logistic_regression')
```

## Model Performance

The system evaluates model performance using:
- Precision, recall, and F1 score
- ROC curve and AUC
- Precision-recall curve
- Confusion matrix
- Feature importance analysis
- SHAP values for model interpretability

## License

This project is licensed under the terms of the license included in the repository.

## Contributing

Contributions to improve the algorithm or extend its capabilities are welcome. Please feel free to submit a pull request.

## Disclaimer

This system is designed for educational and illustrative purposes. Deployment in a production environment would require additional features, security measures, and compliance considerations specific to your regulatory environment.
