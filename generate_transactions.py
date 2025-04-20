import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import pycountry

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

def generate_transactions(customer_profiles_file, num_transactions=10000):
    """
    Generate synthetic transactions for AML analysis
    
    Args:
        customer_profiles_file: Path to the customer profiles CSV file
        num_transactions: Number of transactions to generate
        
    Returns:
        DataFrame containing synthetic transactions
    """
    # Load customer profiles
    customers_df = pd.read_csv(customer_profiles_file)
    customer_ids = customers_df['customer_id'].tolist()
    
    # Get all country codes for cross-border transactions
    country_codes = [country.alpha_2 for country in list(pycountry.countries)]
    
    # Generate transaction IDs
    transaction_ids = [f"TXN{i:08d}" for i in range(1, num_transactions + 1)]
    
    # Choose customer IDs (some customers will have more transactions than others)
    # Using exponential distribution to simulate that some customers are more active
    customer_indices = np.random.exponential(scale=len(customer_ids)/3, size=num_transactions).astype(int)
    customer_indices = np.clip(customer_indices, 0, len(customer_ids) - 1)
    transaction_customer_ids = [customer_ids[i] for i in customer_indices]
    
    # Generate transaction dates (in the past 3 years)
    current_date = datetime.now()
    max_days_ago = 365 * 3  # 3 years
    
    transaction_dates = []
    for _ in range(num_transactions):
        days_ago = random.randint(0, max_days_ago)
        minutes_ago = random.randint(0, 24 * 60)  # Random time during the day
        transaction_date = current_date - timedelta(days=days_ago, minutes=minutes_ago)
        transaction_dates.append(transaction_date.strftime('%Y-%m-%d %H:%M:%S'))
    
    # Sort transactions by date (oldest first)
    transaction_dates = sorted(transaction_dates)
    
    # Generate transaction amounts (most are small, some are large)
    # Using a mixture of normal distributions
    small_amounts = np.random.normal(loc=200, scale=150, size=int(num_transactions * 0.7))
    medium_amounts = np.random.normal(loc=1000, scale=500, size=int(num_transactions * 0.2))
    large_amounts = np.random.normal(loc=8000, scale=3000, size=num_transactions - len(small_amounts) - len(medium_amounts))
    
    all_amounts = np.concatenate([small_amounts, medium_amounts, large_amounts])
    np.random.shuffle(all_amounts)
    # Ensure all amounts are positive and round to 2 decimal places
    transaction_amounts = np.round(np.maximum(all_amounts, 10), 2)
    
    # Generate transaction types
    transaction_types = np.random.choice(
        ["deposit", "withdrawal", "transfer", "payment"],
        size=num_transactions,
        p=[0.25, 0.20, 0.35, 0.20]
    )
    
    # Generate cash transaction flags (more likely for deposits and withdrawals)
    is_cash_transaction = []
    for txn_type in transaction_types:
        if txn_type == "deposit":
            is_cash = np.random.choice([0, 1], p=[0.3, 0.7])
        elif txn_type == "withdrawal":
            is_cash = np.random.choice([0, 1], p=[0.2, 0.8])
        else:
            is_cash = np.random.choice([0, 1], p=[0.9, 0.1])
        is_cash_transaction.append(is_cash)
    
    # Generate cross-border flags (relatively less common)
    is_cross_border = np.random.choice([0, 1], size=num_transactions, p=[0.8, 0.2])
    
    # Generate country codes
    # For domestic transactions, use a default country
    default_country = "US"
    transaction_country_codes = []
    
    for is_cross in is_cross_border:
        if is_cross == 1:
            # For cross-border, choose a random country
            country = np.random.choice(country_codes)
            transaction_country_codes.append(country)
        else:
            transaction_country_codes.append(default_country)
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'customer_id': transaction_customer_ids,
        'transaction_date': transaction_dates,
        'amount': transaction_amounts,
        'transaction_type': transaction_types,
        'is_cash_transaction': is_cash_transaction,
        'is_cross_border': is_cross_border,
        'country_code': transaction_country_codes
    })
    
    # Sort by transaction date
    df = df.sort_values(by='transaction_date').reset_index(drop=True)
    
    return df

def generate_country_risk_ratings():
    """Generate country risk ratings for AML analysis"""
    # Get all countries
    countries = list(pycountry.countries)
    country_codes = [country.alpha_2 for country in countries]
    country_names = [country.name for country in countries]
    
    # Assign risk ratings (mostly low and medium, few high)
    risk_ratings = np.random.choice(
        ["low", "medium", "high"],
        size=len(country_codes),
        p=[0.7, 0.2, 0.1]
    )
    
    # Create DataFrame
    df = pd.DataFrame({
        'country_code': country_codes,
        'country_name': country_names,
        'risk_rating': risk_ratings
    })
    
    return df

if __name__ == "__main__":
    # Generate transactions
    print("Generating transaction data...")
    customer_profiles_path = 'customer_profiles.csv'
    transactions_df = generate_transactions(customer_profiles_path, num_transactions=10000)
    
    # Save transactions to CSV
    transactions_output_path = 'transactions.csv'
    transactions_df.to_csv(transactions_output_path, index=False)
    print(f"Transaction data saved to {transactions_output_path}")
    
    # Display transaction statistics
    print("\nTransaction statistics:")
    print(f"Total transactions: {len(transactions_df)}")
    print(f"Unique customers: {transactions_df['customer_id'].nunique()}")
    print(f"Transaction types distribution:\n{transactions_df['transaction_type'].value_counts()}")
    print(f"Cash transactions: {transactions_df['is_cash_transaction'].sum()}")
    print(f"Cross-border transactions: {transactions_df['is_cross_border'].sum()}")
    
    # Generate and save country risk ratings
    print("\nGenerating country risk ratings...")
    country_risk_df = generate_country_risk_ratings()
    country_risk_path = 'country_risk_ratings.csv'
    country_risk_df.to_csv(country_risk_path, index=False)
    print(f"Country risk ratings saved to {country_risk_path}")