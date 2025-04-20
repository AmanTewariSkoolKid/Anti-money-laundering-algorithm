import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_labeled_cases(customer_profiles_file, transactions_file, ratio_suspicious=0.15):
    """
    Generate labeled cases for AML model training
    
    Args:
        customer_profiles_file: Path to customer profiles CSV
        transactions_file: Path to transactions CSV
        ratio_suspicious: Proportion of cases to label as suspicious
        
    Returns:
        DataFrame with labeled cases
    """
    # Read customer profiles and transactions
    customers_df = pd.read_csv(customer_profiles_file)
    transactions_df = pd.read_csv(transactions_file)
    
    # IMPORTANT: Only use customer IDs from the customer_profiles file to ensure consistency
    customer_ids = customers_df['customer_id'].unique()
    
    # Create a risk score calculation based on:
    # 1. Customer profile factors (PEP status, sanctions match, address changes)
    # 2. Transaction patterns (high amounts, cash transactions, cross-border)
    
    # Calculate transaction stats per customer
    customer_transaction_stats = transactions_df.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'max', 'std'],
        'is_cash_transaction': 'sum',
        'is_cross_border': 'sum',
        'transaction_id': 'count'
    })
    
    customer_transaction_stats.columns = [
        'total_amount', 'avg_amount', 'max_amount', 'std_amount',
        'num_cash_txns', 'num_cross_border_txns', 'total_txns'
    ]
    
    # Reset index to convert customer_id back to a column
    customer_transaction_stats = customer_transaction_stats.reset_index()
    
    # Merge with customer profiles
    risk_df = pd.merge(
        customers_df[['customer_id', 'pep_status', 'sanctions_match', 'address_change_count']],
        customer_transaction_stats,
        on='customer_id',
        how='left'
    )
    
    # Fill NA values for customers with no transactions
    risk_df = risk_df.fillna({
        'total_amount': 0,
        'avg_amount': 0,
        'max_amount': 0,
        'std_amount': 0,
        'num_cash_txns': 0,
        'num_cross_border_txns': 0,
        'total_txns': 0
    })
    
    # Calculate risk scores using a weighted approach
    risk_df['risk_score'] = (
        # Profile risk factors
        risk_df['pep_status'] * 0.3 +
        risk_df['sanctions_match'] * 0.4 +
        risk_df['address_change_count'] * 0.05 +
        
        # Transaction risk factors (normalized)
        np.minimum(risk_df['total_amount'] / 50000, 1) * 0.05 +
        np.minimum(risk_df['avg_amount'] / 5000, 1) * 0.05 +
        np.minimum(risk_df['max_amount'] / 10000, 1) * 0.05 +
        (risk_df['num_cash_txns'] / np.maximum(risk_df['total_txns'], 1)) * 0.05 +
        (risk_df['num_cross_border_txns'] / np.maximum(risk_df['total_txns'], 1)) * 0.05
    )
    
    # Determine how many customers should be labeled suspicious
    num_suspicious = int(len(customer_ids) * ratio_suspicious)
    
    # Get top N customers by risk score
    suspicious_customers = risk_df.sort_values('risk_score', ascending=False)['customer_id'].iloc[:num_suspicious]
    
    # Create labels (1 for suspicious, 0 for normal)
    risk_df['is_suspicious'] = np.where(risk_df['customer_id'].isin(suspicious_customers), 1, 0)
    
    # Generate case dates (in the last year)
    current_date = datetime.now()
    max_days_ago = 365
    
    case_dates = []
    for _ in range(len(risk_df)):
        days_ago = random.randint(0, max_days_ago)
        case_date = current_date - timedelta(days=days_ago)
        case_dates.append(case_date.strftime('%Y-%m-%d'))
    
    risk_df['case_date'] = case_dates
    
    # Create final labeled cases DataFrame with only required columns
    labeled_cases = risk_df[['customer_id', 'is_suspicious', 'case_date']]
    
    return labeled_cases

if __name__ == "__main__":
    print("Generating labeled cases...")
    customer_profiles_path = 'customer_profiles.csv'
    transactions_path = 'transactions.csv'
    
    labeled_cases_df = generate_labeled_cases(customer_profiles_path, transactions_path)
    
    # Save to CSV
    labeled_cases_path = 'labeled_cases.csv'
    labeled_cases_df.to_csv(labeled_cases_path, index=False)
    print(f"Labeled cases saved to {labeled_cases_path}")
    
    # Display statistics
    print("\nLabeled cases statistics:")
    print(f"Total cases: {len(labeled_cases_df)}")
    print(f"Suspicious cases: {labeled_cases_df['is_suspicious'].sum()}")
    print(f"Normal cases: {len(labeled_cases_df) - labeled_cases_df['is_suspicious'].sum()}")
    print(f"Suspicious ratio: {labeled_cases_df['is_suspicious'].mean():.2%}")