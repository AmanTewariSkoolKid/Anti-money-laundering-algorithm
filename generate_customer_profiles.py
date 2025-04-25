import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import pycountry

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)
fake = Faker()
Faker.seed(42)

def generate_customer_profiles(num_customers=1000):
    """
    Generate synthetic customer profiles for AML analysis
    
    Args:
        num_customers: Number of customer profiles to generate
        
    Returns:
        DataFrame containing synthetic customer profiles
    """
    # Get list of all country names
    countries = [country.name for country in list(pycountry.countries)]
    
    # Generate customer IDs
    customer_ids = [f"CUST{i:06d}" for i in range(1, num_customers + 1)]
    
    # Generate ages between 18 and 90
    ages = np.random.randint(18, 91, size=num_customers)
    
    # Generate random nationalities from real countries
    nationalities = np.random.choice(countries, size=num_customers)
    
    # Common occupation categories
    occupations = [
        "Engineer", "Doctor", "Teacher", "Student", "Business Owner", 
        "Accountant", "Lawyer", "Sales Representative", "Manager", "Retired",
        "Consultant", "Government Employee", "Researcher", "IT Professional", 
        "Financial Analyst", "Unemployed"
    ]
    customer_occupations = np.random.choice(occupations, size=num_customers)
    
    # Generate account opening dates (between 1 and 15 years ago)
    current_date = datetime.now()
    max_days_ago = 365 * 15  # 15 years
    min_days_ago = 365 * 1   # 1 year
    
    account_opening_dates = []
    for _ in range(num_customers):
        days_ago = random.randint(min_days_ago, max_days_ago)
        account_date = current_date - timedelta(days=days_ago)
        account_opening_dates.append(account_date.strftime('%Y-%m-%d'))
    
    # Generate PEP status (relatively rare - about 2% of customers)
    pep_status = np.random.choice([0, 1], size=num_customers, p=[0.98, 0.02])
    
    # Generate sanctions match (very rare - about 0.5% of customers)
    sanctions_match = np.random.choice([0, 1], size=num_customers, p=[0.995, 0.005])
    
    # Generate address change counts (most have 0-2, some have more)
    address_change_count = np.random.choice(
        [0, 1, 2, 3, 4, 5], 
        size=num_customers, 
        p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03]
    )
    
    # Generate names using Faker
    names = [fake.name() for _ in range(num_customers)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'name': names,
        'age': ages,
        'nationality': nationalities,
        'occupation': customer_occupations,
        'account_opening_date': account_opening_dates,
        'pep_status': pep_status,
        'sanctions_match': sanctions_match,
        'address_change_count': address_change_count
    })
    
    return df

if __name__ == "__main__":
    # Create data_csv folder if it doesn't exist
    import os
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data_csv')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Generating customer profiles...")
    customer_df = generate_customer_profiles(num_customers=1000)
    
    # Save to CSV
    output_path = os.path.join(data_dir, 'customer_profiles.csv')
    customer_df.to_csv(output_path, index=False)
    print(f"Customer profiles saved to {output_path}")
    
    # Display sample
    print("\nSample customer profiles:")
    print(customer_df.head())
    
    # Display statistics
    print("\nCustomer profile statistics:")
    print(f"Total customers: {len(customer_df)}")
    print(f"PEP customers: {customer_df['pep_status'].sum()}")
    print(f"Sanctioned customers: {customer_df['sanctions_match'].sum()}")