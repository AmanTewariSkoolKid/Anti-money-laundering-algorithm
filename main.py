import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer
import shap
import joblib
from datetime import datetime

class AMLModel:
    def __init__(self, data_path=None, model_name='aml_model'):
        """
        Initialize the AML model
        
        Args:
            data_path: Path to data directory
            model_name: Name for the trained model
        """
        # Use the directory where this script is located if no path provided
        if data_path is None:
            self.data_path = os.path.dirname(os.path.abspath(__file__))
        else:
            self.data_path = data_path
            
        self.model_name = model_name
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None

    def validate_data_consistency(self, fix_inconsistencies=True):
        """Validate that primary and foreign keys are consistent across all files
        
        Args:
            fix_inconsistencies: If True, attempt to fix any inconsistencies found
            
        Returns:
            Boolean indicating if data is consistent (after fixes if applicable)
        """
        print("\nValidating data consistency...")
        
        # Load all datasets
        customer_profiles_path = os.path.join(self.data_path, 'customer_profiles.csv')
        transactions_path = os.path.join(self.data_path, 'transactions.csv')
        country_risk_path = os.path.join(self.data_path, 'country_risk_ratings.csv')
        labeled_cases_path = os.path.join(self.data_path, 'labeled_cases.csv')
        
        try:
            customers_df = pd.read_csv(customer_profiles_path)
            transactions_df = pd.read_csv(transactions_path)
            country_risk_df = pd.read_csv(country_risk_path)
            labeled_cases_df = pd.read_csv(labeled_cases_path)
            
            # Check 1: All customer IDs in transactions exist in customer profiles
            transaction_customers = set(transactions_df['customer_id'].unique())
            profile_customers = set(customers_df['customer_id'].unique())
            invalid_transaction_customers = transaction_customers - profile_customers
            
            if invalid_transaction_customers:
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
                    print("Please regenerate your data files for consistency.")
                    return False
            else:
                print("OK: All transaction customer IDs exist in customer profiles")
            
            # Check 2: All customer IDs in labeled cases exist in customer profiles
            labeled_customers = set(labeled_cases_df['customer_id'].unique())
            invalid_labeled_customers = labeled_customers - profile_customers
            
            if invalid_labeled_customers:
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
                    print("Please regenerate your data files for consistency.")
                    return False
            else:
                print("OK: All labeled case customer IDs exist in customer profiles")
            
            # Check 3: All country codes in transactions exist in country risk ratings
            # Need to reload transactions if they were modified
            if invalid_transaction_customers and fix_inconsistencies:
                transactions_df = pd.read_csv(transactions_path)
                
            transaction_countries = set(transactions_df['country_code'].unique())
            risk_countries = set(country_risk_df['country_code'].unique())
            invalid_transaction_countries = transaction_countries - risk_countries
            
            if invalid_transaction_countries:
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
                    print("Please regenerate your data files for consistency.")
                    return False
            else:
                print("OK: All transaction country codes exist in country risk ratings")
                
            print("Data consistency validation complete")
            return True
            
        except FileNotFoundError as e:
            print(f"Error: Required data file not found. {str(e)}")
            print("Please make sure all required CSV files exist in the data directory.")
            return False
        except Exception as e:
            print(f"Error validating data consistency: {str(e)}")
            return False
    
    def load_data(self):
        """Load and prepare data for model training"""
        print("\nLoading and preparing data...")
        
        # Check if required files exist
        required_files = ['customer_profiles.csv', 'transactions.csv', 
                         'country_risk_ratings.csv', 'labeled_cases.csv']
        for file in required_files:
            file_path = os.path.join(self.data_path, file)
            if not os.path.exists(file_path):
                print(f"Error: Required file '{file}' not found in {self.data_path}")
                print("Please generate all required data files first.")
                return False
        
        # Validate data consistency with auto-fix enabled
        if not self.validate_data_consistency(fix_inconsistencies=True):
            return False
        
        # Load CSVs
        customers_df = pd.read_csv(os.path.join(self.data_path, 'customer_profiles.csv'))
        transactions_df = pd.read_csv(os.path.join(self.data_path, 'transactions.csv'))
        country_risk_df = pd.read_csv(os.path.join(self.data_path, 'country_risk_ratings.csv'))
        labeled_cases_df = pd.read_csv(os.path.join(self.data_path, 'labeled_cases.csv'))
        
        # Merge country risk ratings with transactions
        transactions_with_risk = pd.merge(
            transactions_df,
            country_risk_df[['country_code', 'risk_rating']],
            on='country_code',
            how='left'
        )
        
        # Create risk rating numerical representation
        risk_mapping = {'low': 0, 'medium': 1, 'high': 2}
        transactions_with_risk['country_risk_score'] = transactions_with_risk['risk_rating'].map(risk_mapping).fillna(0)
        
        # Calculate aggregated transaction features per customer
        transaction_features = self._calculate_transaction_features(transactions_with_risk)
        
        # Merge all data sources
        customer_features = pd.merge(
            customers_df,
            transaction_features,
            on='customer_id',
            how='left'
        )
        
        # Fill NA values for customers with no transactions
        customer_features = self._fill_missing_transaction_features(customer_features)
        
        # Add labels
        final_dataset = pd.merge(
            customer_features,
            labeled_cases_df[['customer_id', 'is_suspicious']],
            on='customer_id',
            how='inner'
        )
        
        # Separate features and target
        X = final_dataset.drop(['customer_id', 'name', 'is_suspicious', 'account_opening_date'], axis=1)
        y = final_dataset['is_suspicious']
        
        # Store feature names for later
        self.feature_names = X.columns.tolist()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        print(f"Data loaded and prepared. Training set size: {len(self.X_train)}, Test set size: {len(self.X_test)}")
        print(f"Feature count: {len(self.feature_names)}")
        print(f"Class distribution in training set: {dict(zip(*np.unique(self.y_train, return_counts=True)))}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def _calculate_transaction_features(self, transactions_df):
        """Calculate aggregated transaction features per customer"""
        # First, calculate individual transaction type counts
        # This avoids the issue with Series objects in a DataFrame cell
        transaction_counts = transactions_df.groupby('customer_id')['transaction_type'].value_counts().unstack().fillna(0)
        
        # Ensure all transaction types have columns, even if they're all zeros
        for txn_type in ['deposit', 'withdrawal', 'transfer', 'payment']:
            if txn_type not in transaction_counts.columns:
                transaction_counts[txn_type] = 0
        
        # Rename for clarity
        transaction_counts.columns = [f'{col}_count' for col in transaction_counts.columns]
        transaction_counts = transaction_counts.reset_index()
        
        # Group transactions by customer_id for regular aggregations
        transaction_aggs = transactions_df.groupby('customer_id').agg({
            # Amount-based features
            'amount': ['sum', 'mean', 'std', 'max', 'count'],
            
            # Cash transaction features
            'is_cash_transaction': ['sum', 'mean'],
            
            # Cross-border features
            'is_cross_border': ['sum', 'mean'],
            
            # Country risk features
            'country_risk_score': ['max', 'mean']
        })
        
        # Flatten multi-level column names
        transaction_aggs.columns = ['_'.join(col).strip() for col in transaction_aggs.columns.values]
        
        # Rename columns for clarity
        transaction_aggs = transaction_aggs.rename(columns={
            'amount_sum': 'total_transaction_amount',
            'amount_mean': 'avg_transaction_amount',
            'amount_std': 'std_transaction_amount',
            'amount_max': 'max_transaction_amount',
            'amount_count': 'transaction_count',
            'is_cash_transaction_sum': 'cash_transaction_count',
            'is_cash_transaction_mean': 'cash_transaction_ratio',
            'is_cross_border_sum': 'cross_border_count',
            'is_cross_border_mean': 'cross_border_ratio',
            'country_risk_score_max': 'max_country_risk',
            'country_risk_score_mean': 'avg_country_risk'
        })
        
        # Reset index to make customer_id a column
        transaction_aggs = transaction_aggs.reset_index()
        
        # Calculate additional derived features
        # Ratio of cash to total transaction amount (if available)
        cash_transactions = transactions_df[transactions_df['is_cash_transaction'] == 1]
        cash_by_customer = cash_transactions.groupby('customer_id')['amount'].sum().reset_index()
        cash_by_customer = cash_by_customer.rename(columns={'amount': 'total_cash_amount'})
        
        # Merge all the features into one DataFrame
        transaction_features = pd.merge(transaction_aggs, transaction_counts, on='customer_id', how='left')
        transaction_features = pd.merge(transaction_features, cash_by_customer, on='customer_id', how='left')
        
        # Fill NAs and calculate ratios
        transaction_features['total_cash_amount'] = transaction_features['total_cash_amount'].fillna(0)
        transaction_features['cash_amount_ratio'] = transaction_features['total_cash_amount'] / transaction_features['total_transaction_amount']
        transaction_features['cash_amount_ratio'] = transaction_features['cash_amount_ratio'].fillna(0)
        
        return transaction_features
    
    def _fill_missing_transaction_features(self, df):
        """Fill missing transaction features with zeros"""
        transaction_cols = [col for col in df.columns if col not in 
                           ['customer_id', 'name', 'age', 'nationality', 
                            'occupation', 'account_opening_date', 'pep_status', 
                            'sanctions_match', 'address_change_count']]
        
        return df.fillna({col: 0 for col in transaction_cols})
    
    def train_model(self, model_type='random_forest'):
        """Train the ML model for AML detection
        
        Args:
            model_type: Type of model to train ('random_forest', 'gradient_boosting', or 'logistic_regression')
            
        Returns:
            Trained model
        """
        print("\nTraining AML detection model...")
        
        if self.X_train is None or self.y_train is None:
            print("Error: Data not loaded. Call load_data() first.")
            return None
        
        # Debug: Check for problematic data types or values
        print("Checking data types and cleaning data...")
        for col in self.X_train.columns:
            # Check if column contains lists or other non-scalar values
            if any(isinstance(x, (list, dict, tuple)) for x in self.X_train[col].dropna().values):
                print(f"WARNING: Column '{col}' contains non-scalar values. Converting to strings.")
                self.X_train[col] = self.X_train[col].astype(str)
                self.X_test[col] = self.X_test[col].astype(str)
            
            # Check for NaN values
            if self.X_train[col].isna().any():
                print(f"WARNING: Column '{col}' contains NaN values. Will be handled by imputer.")
                
            # Force numeric columns to be float
            if pd.api.types.is_numeric_dtype(self.X_train[col]):
                self.X_train[col] = self.X_train[col].astype(float)
                self.X_test[col] = self.X_test[col].astype(float)
                
        # Define categorical and numerical columns based on actual data types
        categorical_cols = [col for col in self.feature_names 
                           if not pd.api.types.is_numeric_dtype(self.X_train[col])]
        numerical_cols = [col for col in self.feature_names 
                         if pd.api.types.is_numeric_dtype(self.X_train[col])]
        
        print(f"Categorical columns: {categorical_cols}")
        print(f"Numerical columns: {numerical_cols}")
        
        # Create preprocessing pipeline
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        
        # Store preprocessor for later use with new data
        self.preprocessor = preprocessor
        
        # Select and configure the model
        if model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=None,
                min_samples_split=10,
                min_samples_leaf=4,
                max_features='sqrt',
                class_weight='balanced',
                random_state=42
            )
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__max_depth': [None, 10, 20],
                'model__min_samples_split': [5, 10]
            }
        elif model_type == 'gradient_boosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                random_state=42
            )
            param_grid = {
                'model__n_estimators': [50, 100],
                'model__learning_rate': [0.05, 0.1],
                'model__max_depth': [3, 5]
            }
        elif model_type == 'logistic_regression':
            model = LogisticRegression(
                C=1.0,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )
            param_grid = {
                'model__C': [0.1, 1.0, 10.0],
                'model__solver': ['liblinear', 'saga']
            }
        else:
            print(f"Error: Unknown model type '{model_type}'")
            return None
            
        # Create full pipeline
        full_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        # Set up cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        # Set up grid search with error debugging enabled
        grid_search = GridSearchCV(
            full_pipeline, 
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=2,
            error_score='raise'  # Raise errors during fitting for better debugging
        )
        
        # Fit the model
        print(f"Performing grid search for {model_type}...")
        try:
            grid_search.fit(self.X_train, self.y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            
            # Save the model to disk
            model_path = f"{self.model_name}_{model_type}.pkl"
            joblib.dump(self.model, model_path)
            print(f"Model saved to {model_path}")
            print(f"Best parameters: {grid_search.best_params_}")
            
            return self.model
            
        except Exception as e:
            print(f"Error during model training: {str(e)}")
            print("Trying simplified model without grid search...")
            
            # Try a simpler model without grid search
            try:
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
                pipeline.fit(self.X_train, self.y_train)
                self.model = pipeline
                
                # Save the model
                model_path = f"{self.model_name}_{model_type}_simple.pkl"
                joblib.dump(self.model, model_path)
                print(f"Simple model saved to {model_path}")
                
                return self.model
            except Exception as e2:
                print(f"Error training simplified model: {str(e2)}")
                return None
    
    def evaluate_model(self):
        """Evaluate the trained model and return performance metrics"""
        if self.model is None:
            print("Error: Model not trained. Call train_model() first.")
            return None
            
        print("\nEvaluating model performance...")
            
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        precision = precision_score(self.y_test, y_pred)
        recall = recall_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred)
        
        # Create classification report
        class_report = classification_report(self.y_test, y_pred)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Calculate Precision-Recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(self.y_test, y_pred_proba)
        pr_auc = auc(recall_curve, precision_curve)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Prepare results dictionary
        results = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'fpr': fpr,
            'tpr': tpr,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve
        }
        
        return results
        
    def visualize_results(self, results):
        """Create visualizations for model evaluation results"""
        if results is None:
            print("Error: No results to visualize.")
            return
            
        # Create output directory if it doesn't exist - using absolute path
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        print(f"Saving results to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp for unique filenames
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. Confusion Matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=['Not Suspicious', 'Suspicious'],
            yticklabels=['Not Suspicious', 'Suspicious']
        )
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        confusion_path = os.path.join(output_dir, f'confusion_matrix_{timestamp}.png')
        plt.savefig(confusion_path)
        plt.close()
        
        # 2. ROC Curve
        plt.figure(figsize=(8, 6))
        plt.plot(
            results['fpr'], 
            results['tpr'], 
            lw=2, 
            label=f"ROC curve (AUC = {results['roc_auc']:.2f})"
        )
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        roc_path = os.path.join(output_dir, f'roc_curve_{timestamp}.png')
        plt.savefig(roc_path)
        plt.close()
        
        # 3. Precision-Recall Curve
        plt.figure(figsize=(8, 6))
        plt.plot(
            results['recall_curve'], 
            results['precision_curve'], 
            lw=2, 
            label=f"PR curve (AUC = {results['pr_auc']:.2f})"
        )
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        pr_path = os.path.join(output_dir, f'pr_curve_{timestamp}.png')
        plt.savefig(pr_path)
        plt.close()
        
        # 4. Feature Importance (if model supports it)
        if hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['model'], 'feature_importances_'):
            # For tree-based models
            feature_names = self.feature_names.copy()
            
            # Get feature names from the preprocessor
            if hasattr(self.model.named_steps['preprocessor'], 'transformers_'):
                all_features = []
                for name, trans, cols in self.model.named_steps['preprocessor'].transformers_:
                    if name == 'cat' and hasattr(trans, 'named_steps') and hasattr(trans.named_steps['onehot'], 'get_feature_names_out'):
                        # For categorical features, get the one-hot encoded feature names
                        encoded_features = trans.named_steps['onehot'].get_feature_names_out(cols)
                        all_features.extend(encoded_features)
                    else:
                        # For numerical features, use the original names
                        all_features.extend(cols)
                        
                feature_names = all_features
            
            # Get feature importances
            importances = self.model.named_steps['model'].feature_importances_
            
            # Sort features by importance and take top 20
            indices = np.argsort(importances)[::-1][:20]
            
            # Plot feature importance
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in indices])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 Feature Importance')
            feat_path = os.path.join(output_dir, f'feature_importance_{timestamp}.png')
            plt.savefig(feat_path)
            plt.close()
            
        # 5. SHAP values for feature importance explanation
        try:
            # Create a simplified version of our pipeline for SHAP
            # We need to preprocess the test data and get the feature names
            X_test_processed = self.model.named_steps['preprocessor'].transform(self.X_test)
            
            # Create a SHAP explainer for the model
            if hasattr(self.model.named_steps['model'], 'predict_proba'):
                # For tree-based models
                if hasattr(self.model.named_steps['model'], 'estimators_'):
                    explainer = shap.TreeExplainer(self.model.named_steps['model'])
                else:
                    explainer = shap.Explainer(self.model.named_steps['model'].predict_proba, X_test_processed)
                
                # Calculate SHAP values for a sample of test data
                sample_size = min(100, X_test_processed.shape[0])
                sample_indices = np.random.choice(X_test_processed.shape[0], sample_size, replace=False)
                shap_values = explainer.shap_values(X_test_processed[sample_indices])
                
                # Convert to expected type if needed
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # For binary classification, use positive class
                
                # Plot SHAP summary
                plt.figure(figsize=(12, 8))
                shap.summary_plot(
                    shap_values, 
                    X_test_processed[sample_indices],
                    show=False
                )
                plt.title('SHAP Feature Importance')
                shap_path = os.path.join(output_dir, f'shap_importance_{timestamp}.png')
                plt.savefig(shap_path)
                plt.close()
        except Exception as e:
            print(f"Warning: Could not generate SHAP plots. Error: {str(e)}")
        
        # Return the paths of saved visualizations
        visualization_paths = {
            'confusion_matrix': confusion_path,
            'roc_curve': roc_path,
            'pr_curve': pr_path,
            'feature_importance': feat_path if 'feat_path' in locals() else None,
            'shap_importance': shap_path if 'shap_path' in locals() else None
        }
        
        return visualization_paths
        
    def save_results_to_file(self, results, visualization_paths):
        """Save all results to a comprehensive report file"""
        if results is None:
            print("Error: No results to save.")
            return
            
        # Create output directory if it doesn't exist - using absolute path
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
        print(f"Saving results to directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # Timestamp for unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = os.path.join(output_dir, f'aml_detection_results_{timestamp}.txt')
        
        with open(results_file, 'w') as f:
            # Write header
            f.write('=' * 80 + '\n')
            f.write('AML RISK DETECTION MODEL RESULTS\n')
            f.write('=' * 80 + '\n\n')
            
            # Write timestamp
            f.write(f'Report generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
            
            # Write model information
            model_type = type(self.model.named_steps['model']).__name__
            f.write(f'Model type: {model_type}\n\n')
            
            # Write performance metrics
            f.write('-' * 40 + '\n')
            f.write('PERFORMANCE METRICS\n')
            f.write('-' * 40 + '\n\n')
            f.write(f'Precision:    {results["precision"]:.4f}\n')
            f.write(f'Recall:       {results["recall"]:.4f}\n')
            f.write(f'F1 Score:     {results["f1_score"]:.4f}\n')
            f.write(f'ROC AUC:      {results["roc_auc"]:.4f}\n')
            f.write(f'PR AUC:       {results["pr_auc"]:.4f}\n\n')
            
            # Write classification report
            f.write('-' * 40 + '\n')
            f.write('CLASSIFICATION REPORT\n') 
            f.write('-' * 40 + '\n\n')
            f.write(results["classification_report"] + '\n')
            
            # Write confusion matrix
            f.write('-' * 40 + '\n')
            f.write('CONFUSION MATRIX\n')
            f.write('-' * 40 + '\n\n')
            f.write('                  Predicted\n')
            f.write('                  Not Suspicious  Suspicious\n')
            f.write(f'Actual Not Suspicious  {results["confusion_matrix"][0, 0]}            {results["confusion_matrix"][0, 1]}\n')
            f.write(f'      Suspicious      {results["confusion_matrix"][1, 0]}            {results["confusion_matrix"][1, 1]}\n\n')
            
            # Write visualization paths
            f.write('-' * 40 + '\n')
            f.write('VISUALIZATIONS\n')
            f.write('-' * 40 + '\n\n')
            for name, path in visualization_paths.items():
                if path is not None:
                    f.write(f'{name.replace("_", " ").title()}: {path}\n')
            
            # Note about suspicious transactions file
            suspicious_transactions_file = os.path.join(output_dir, f'suspicious_transactions_{timestamp}.csv')
            f.write('\n' + '-' * 40 + '\n')
            f.write('SUSPICIOUS TRANSACTIONS\n')
            f.write('-' * 40 + '\n\n')
            f.write(f'List of flagged suspicious transactions saved to: {suspicious_transactions_file}\n')
            
        print(f"\nResults saved to: {results_file}")
        
        # Also save the suspicious transactions details
        self.save_suspicious_transactions(timestamp, output_dir)
        
        return results_file
        
    def save_suspicious_transactions(self, timestamp, output_dir):
        """Save detailed information about suspicious transactions flagged by the model"""
        try:
            print("\nGenerating suspicious transactions report...")
            
            # Load transaction data with proper path
            transactions_path = os.path.join(self.data_path, 'transactions.csv')
            print(f"Loading transactions from: {transactions_path}")
            transactions_df = pd.read_csv(transactions_path)
            
            # Load customer profiles with proper path
            customers_path = os.path.join(self.data_path, 'customer_profiles.csv')
            print(f"Loading customer profiles from: {customers_path}")
            customers_df = pd.read_csv(customers_path)
            
            # Load country risk ratings with proper path
            risk_path = os.path.join(self.data_path, 'country_risk_ratings.csv')
            print(f"Loading country risk ratings from: {risk_path}")
            country_risk_df = pd.read_csv(risk_path)
            
            # Merge country risk ratings with transactions
            print("Merging country risk data with transactions...")
            transactions_with_risk = pd.merge(
                transactions_df,
                country_risk_df[['country_code', 'risk_rating']],
                on='country_code',
                how='left'
            )
            
            # Create risk rating numerical representation
            risk_mapping = {'low': 0, 'medium': 1, 'high': 2}
            transactions_with_risk['country_risk_score'] = transactions_with_risk['risk_rating'].map(risk_mapping).fillna(1)
            
            # Calculate aggregated transaction features per customer
            print("Calculating customer transaction features...")
            transaction_features = self._calculate_transaction_features(transactions_with_risk)
            
            # Merge all data sources for all customers
            all_customer_features = pd.merge(
                customers_df,
                transaction_features,
                on='customer_id',
                how='left'
            )
            
            # Fill NA values for customers with no transactions
            all_customer_features = self._fill_missing_transaction_features(all_customer_features)
            
            # Get features for prediction (drop non-feature columns)
            print("Preparing data for model prediction...")
            X_all = all_customer_features.drop(['customer_id', 'name', 'account_opening_date'], axis=1)
            
            # Handle any data type issues
            for col in X_all.columns:
                # Force numeric columns to be float
                if pd.api.types.is_numeric_dtype(X_all[col]):
                    X_all[col] = X_all[col].fillna(0).astype(float)
                else:
                    X_all[col] = X_all[col].fillna("unknown").astype(str)
                    
            # Make predictions for all customers
            print("Making predictions for all customers...")
            y_pred = self.model.predict(X_all)
            y_prob = self.model.predict_proba(X_all)[:, 1]  # Probability of being suspicious
            
            # Add predictions to customer data
            all_customer_features['predicted_suspicious'] = y_pred
            all_customer_features['suspicious_probability'] = y_prob
            
            # Filter only suspicious customers (predicted as 1)
            print("Filtering suspicious customers...")
            suspicious_customers = all_customer_features[all_customer_features['predicted_suspicious'] == 1].copy()
            
            # Check if we found any suspicious customers
            if len(suspicious_customers) == 0:
                print("No suspicious customers were identified by the model.")
                return None
                
            print(f"Found {len(suspicious_customers)} suspicious customers")
            
            # Get all transactions for suspicious customers
            print("Extracting transactions for suspicious customers...")
            suspicious_transactions = pd.merge(
                transactions_with_risk,
                suspicious_customers[['customer_id', 'name', 'suspicious_probability']],
                on='customer_id',
                how='inner'
            )
            
            # Add a risk score based on transaction characteristics
            print("Calculating transaction risk scores...")
            suspicious_transactions['transaction_risk_score'] = (
                suspicious_transactions['amount'] / 1000 +  # Higher amounts are riskier
                suspicious_transactions['is_cash_transaction'] * 3 +  # Cash transactions are high risk
                suspicious_transactions['is_cross_border'] * 2 +  # Cross-border transactions are risky
                suspicious_transactions['country_risk_score']  # Country risk factor
            )
            
            # Sort by risk indicators (highest risk first)
            suspicious_transactions = suspicious_transactions.sort_values(
                by=['suspicious_probability', 'transaction_risk_score'], 
                ascending=[False, False]
            )
            
            # Save the suspicious transactions to CSV
            suspicious_file = os.path.join(output_dir, f'suspicious_transactions_{timestamp}.csv')
            print(f"Saving suspicious transactions to: {suspicious_file}")
            suspicious_transactions.to_csv(suspicious_file, index=False)
            
            # Create a summary file with suspicious customers
            customer_summary = suspicious_customers[['customer_id', 'name', 'age', 'nationality', 
                                                  'occupation', 'pep_status', 'sanctions_match',
                                                  'suspicious_probability']]
            customer_summary['transaction_count'] = suspicious_customers['transaction_count']
            customer_summary['total_amount'] = suspicious_customers['total_transaction_amount']
            
            # Save the suspicious customers to CSV
            customer_file = os.path.join(output_dir, f'suspicious_customers_{timestamp}.csv')
            print(f"Saving suspicious customer summary to: {customer_file}")
            customer_summary.to_csv(customer_file, index=False)
            
            print(f"Suspicious transactions analysis complete!")
            return suspicious_file
            
        except Exception as e:
            print(f"Error generating suspicious transactions report: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
        
if __name__ == "__main__":
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    print("=" * 60)
    print("AML Risk Detection using Machine Learning")
    print("=" * 60)
    
    # Create AML model
    aml_model = AMLModel()
    
    # Validate data consistency
    if not aml_model.validate_data_consistency():
        print("Error: Data consistency issues found. Please fix the issues and try again.")
        exit(1)
    
    # Load and prepare data
    if not aml_model.load_data():
        print("Error: Failed to load data. Please check the data files and try again.")
        exit(1)
        
    # Train the model (random forest by default)
    model = aml_model.train_model(model_type='random_forest')
    if model is None:
        print("Error: Failed to train the model.")
        exit(1)
        
    # Evaluate the model
    results = aml_model.evaluate_model()
    if results is None:
        print("Error: Failed to evaluate the model.")
        exit(1)
        
    # Generate visualizations
    viz_paths = aml_model.visualize_results(results)
    if viz_paths is None:
        print("Warning: Could not generate result visualizations.")
        
    # Save comprehensive results to file
    results_file = aml_model.save_results_to_file(results, viz_paths)
    
    print(f"\nAML Risk Detection analysis complete!")
    print(f"Results file: {results_file}")
    print(f"Visualizations directory: {os.path.join(aml_model.data_path, 'results')}")