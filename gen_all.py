import os
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# Import the data generation modules
from generate_customer_profiles import generate_customer_profiles
from generate_transactions import generate_transactions, generate_country_risk_ratings
from generate_labeled_cases import generate_labeled_cases

def check_data_consistency(data_path=None, fix_inconsistencies=True):
    """
    Check consistency across all generated data files
    
    Args:
        data_path: Path to directory containing data files
        fix_inconsistencies: If True, attempt to fix any inconsistencies
        
    Returns:
        Boolean indicating if data is consistent (after fixes if applicable)
    """
    # Use script directory if data_path is not provided
    if data_path is None:
        data_path = os.path.dirname(os.path.abspath(__file__))
    
    # Look in data_csv folder
    data_path = os.path.join(data_path, 'data_csv')
    
    print("\n" + "="*60)
    print("Validating data consistency...")
    print("="*60)
    
    # Load all datasets
    customer_profiles_path = os.path.join(data_path, 'customer_profiles.csv')
    transactions_path = os.path.join(data_path, 'transactions.csv')
    country_risk_path = os.path.join(data_path, 'country_risk_ratings.csv')
    labeled_cases_path = os.path.join(data_path, 'labeled_cases.csv')
    
    try:
        customers_df = pd.read_csv(customer_profiles_path)
        transactions_df = pd.read_csv(transactions_path)
        country_risk_df = pd.read_csv(country_risk_path)
        labeled_cases_df = pd.read_csv(labeled_cases_path)
        
        issues_found = 0
        
        # Check 1: All customer IDs in transactions exist in customer profiles
        transaction_customers = set(transactions_df['customer_id'].unique())
        profile_customers = set(customers_df['customer_id'].unique())
        invalid_transaction_customers = transaction_customers - profile_customers
        
        if invalid_transaction_customers:
            issues_found += 1
            print(f"WARNING: Found {len(invalid_transaction_customers)} customer IDs in transactions that don't exist in profiles")
            if fix_inconsistencies:
                # Remove transactions with invalid customer IDs
                print("Fixing: Removing transactions with invalid customer IDs...")
                original_count = len(transactions_df)
                transactions_df = transactions_df[transactions_df['customer_id'].isin(profile_customers)]
                removed_count = original_count - len(transactions_df)
                print(f"Removed {removed_count} transactions with invalid customer IDs")
                # Save the filtered data back to CSV
                transactions_df.to_csv(transactions_path, index=False)
            else:
                print("Fix disabled. Data inconsistency remains.")
        else:
            print("+ All transaction customer IDs exist in customer profiles")
        
        # Check 2: All customer IDs in labeled cases exist in customer profiles
        labeled_customers = set(labeled_cases_df['customer_id'].unique())
        invalid_labeled_customers = labeled_customers - profile_customers
        
        if invalid_labeled_customers:
            issues_found += 1
            print(f"WARNING: Found {len(invalid_labeled_customers)} customer IDs in labeled cases that don't exist in profiles")
            if fix_inconsistencies:
                # Remove labeled cases with invalid customer IDs
                print("Fixing: Removing labeled cases with invalid customer IDs...")
                original_count = len(labeled_cases_df)
                labeled_cases_df = labeled_cases_df[labeled_cases_df['customer_id'].isin(profile_customers)]
                removed_count = original_count - len(labeled_cases_df)
                print(f"Removed {removed_count} labeled cases with invalid customer IDs")
                # Save the filtered data back to CSV
                labeled_cases_df.to_csv(labeled_cases_path, index=False)
            else:
                print("Fix disabled. Data inconsistency remains.")
        else:
            print("+ All labeled case customer IDs exist in customer profiles")
        
        # Check 3: All country codes in transactions exist in country risk ratings
        # Need to reload transactions if they were modified
        if invalid_transaction_customers and fix_inconsistencies:
            transactions_df = pd.read_csv(transactions_path)
            
        transaction_countries = set(transactions_df['country_code'].unique())
        risk_countries = set(country_risk_df['country_code'].unique())
        invalid_transaction_countries = transaction_countries - risk_countries
        
        if invalid_transaction_countries:
            issues_found += 1
            print(f"WARNING: Found {len(invalid_transaction_countries)} country codes in transactions that don't exist in risk ratings")
            if fix_inconsistencies:
                # Add missing countries to risk ratings with default "medium" rating
                print("Fixing: Adding missing countries to risk ratings with 'medium' risk level...")
                for country_code in invalid_transaction_countries:
                    new_row = {'country_code': country_code, 
                            'country_name': f"Unknown ({country_code})",
                            'risk_rating': 'medium'}
                    country_risk_df = pd.concat([country_risk_df, pd.DataFrame([new_row])], ignore_index=True)
                
                print(f"Added {len(invalid_transaction_countries)} missing countries to risk ratings")
                # Save the updated data back to CSV
                country_risk_df.to_csv(country_risk_path, index=False)
            else:
                print("Fix disabled. Data inconsistency remains.")
        else:
            print("+ All transaction country codes exist in country risk ratings")
        
        # Check 4: Verify that labeled_cases has sufficient data for modeling
        suspicious_count = labeled_cases_df['is_suspicious'].sum()
        total_count = len(labeled_cases_df)
        suspicious_ratio = suspicious_count / total_count if total_count > 0 else 0
        
        if suspicious_count < 10:
            issues_found += 1
            print(f"WARNING: Only {suspicious_count} suspicious cases found, which might be insufficient for modeling")
            print("Consider regenerating data with a higher suspicious ratio")
        else:
            print(f"+ Found {suspicious_count} suspicious cases ({suspicious_ratio:.1%}) which should be sufficient for modeling")
            
        print("="*60)
        if issues_found == 0:
            print("+ All consistency checks passed!")
            return True
        else:
            if fix_inconsistencies:
                print(f"! {issues_found} issue(s) were found and fixed")
                return True
            else:
                print(f"! {issues_found} issue(s) were found but not fixed")
                return False
            
    except FileNotFoundError as e:
        print(f"Error: Required data file not found. {str(e)}")
        print("Please generate all required data files first.")
        return False
    except Exception as e:
        print(f"Error validating data consistency: {str(e)}")
        return False

def generate_all_files(num_customers=1000, num_transactions=10000, data_path=None):
    """
    Generate all required data files for AML analysis
    
    Args:
        num_customers: Number of customer profiles to generate
        num_transactions: Number of transactions to generate
        data_path: Path to save generated data files
    """
    # Use script directory if data_path is not provided
    if data_path is None:
        data_path = os.path.dirname(os.path.abspath(__file__))
    
    # Create data_csv folder within the data_path
    data_csv_path = os.path.join(data_path, 'data_csv')
    os.makedirs(data_csv_path, exist_ok=True)
    
    print("\n" + "="*60)
    print(f"Generating AML Data Files ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    print("="*60)
    print(f"Saving files to: {data_csv_path}")
    
    # File paths
    customer_profiles_path = os.path.join(data_csv_path, 'customer_profiles.csv')
    transactions_path = os.path.join(data_csv_path, 'transactions.csv')
    country_risk_path = os.path.join(data_csv_path, 'country_risk_ratings.csv')
    labeled_cases_path = os.path.join(data_csv_path, 'labeled_cases.csv')
    
    # Step 1: Generate customer profiles
    print("\nGenerating customer profiles...")
    customer_df = generate_customer_profiles(num_customers)
    customer_df.to_csv(customer_profiles_path, index=False)
    print(f"+ Customer profiles saved to {customer_profiles_path}")
    
    # Step 2: Generate country risk ratings
    print("\nGenerating country risk ratings...")
    country_risk_df = generate_country_risk_ratings()
    country_risk_df.to_csv(country_risk_path, index=False)
    print(f"+ Country risk ratings saved to {country_risk_path}")
    
    # Step 3: Generate transactions
    print("\nGenerating transactions...")
    transactions_df = generate_transactions(customer_profiles_path, num_transactions)
    transactions_df.to_csv(transactions_path, index=False)
    print(f"+ Transactions saved to {transactions_path}")
    
    # Step 4: Generate labeled cases
    print("\nGenerating labeled cases...")
    labeled_cases_df = generate_labeled_cases(customer_profiles_path, transactions_path)
    labeled_cases_df.to_csv(labeled_cases_path, index=False)
    print(f"+ Labeled cases saved to {labeled_cases_path}")
    
    print("\nAll data files generated successfully!")
    
    # Show statistics
    print("\n" + "="*60)
    print("Data Generation Statistics")
    print("="*60)
    print(f"Customer profiles: {len(customer_df)} records")
    print(f"Transactions: {len(transactions_df)} records")
    print(f"Country risk ratings: {len(country_risk_df)} records")
    print(f"Labeled cases: {len(labeled_cases_df)} records")
    print(f"Suspicious cases: {labeled_cases_df['is_suspicious'].sum()} ({labeled_cases_df['is_suspicious'].mean():.1%})")
    
    return {
        'customer_profiles': customer_df,
        'transactions': transactions_df,
        'country_risk_ratings': country_risk_df,
        'labeled_cases': labeled_cases_df
    }

if __name__ == "__main__":
    # Get configuration parameters from user
    try:
        print("\n" + "="*60)
        print("AML Data Generation Configuration")
        print("="*60)
        NUM_CUSTOMERS = int(input("Enter number of customers to generate (recommended: 1000-10000): ").strip() or "10000")
        NUM_TRANSACTIONS = int(input("Enter number of transactions to generate (recommended: 10000-100000): ").strip() or "100000")
        
        print(f"\nGenerating data for {NUM_CUSTOMERS} customers and {NUM_TRANSACTIONS} transactions...")
    except ValueError:
        print("Invalid input. Using default values.")
        NUM_CUSTOMERS = 10000
        NUM_TRANSACTIONS = 100000
    
    # Use the directory where this script is located
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # Generate all files in the same directory as this script
    generate_all_files(NUM_CUSTOMERS, NUM_TRANSACTIONS, SCRIPT_DIR)
    
    # Check consistency after generation
    print("\nChecking consistency of generated files...")
    check_data_consistency(SCRIPT_DIR, fix_inconsistencies=True)
    
    print("\nData generation and consistency check complete!")
    print("You can now run the AML analysis using main.py")