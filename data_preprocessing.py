"""
Data Preprocessing Module for BigFeat
Handles data loading, cleaning, and basic preprocessing
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer


class DataPreprocessor:
    """
    Preprocesses data for BigFeat framework
    """
    
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_name = None
        
    def load_data(self, filepath, target_column):
        """
        Load data from CSV file
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target column
            
        Returns:
            X: Feature matrix
            y: Target vector
        """
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        
        # Separate features and target
        self.target_name = target_column
        y = df[target_column]
        X = df.drop(columns=[target_column])
        
        self.feature_names = X.columns.tolist()
        
        print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def handle_missing_values(self, X):
        """
        Handle missing values using mean imputation
        
        Args:
            X: Feature matrix
            
        Returns:
            X_imputed: Imputed feature matrix
        """
        print("Handling missing values...")
        imputer = SimpleImputer(strategy='mean')
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        missing_count = X.isnull().sum().sum()
        if missing_count > 0:
            print(f"Imputed {missing_count} missing values")
        
        return X_imputed
    
    def encode_categorical_features(self, X):
        """
        Encode categorical features using label encoding
        
        Args:
            X: Feature matrix
            
        Returns:
            X_encoded: Encoded feature matrix
        """
        print("Encoding categorical features...")
        X_encoded = X.copy()
        
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X[col].astype(str))
            print(f"  Encoded column: {col}")
        
        return X_encoded
    
    def encode_target(self, y):
        """
        Encode target variable for classification
        
        Args:
            y: Target vector
            
        Returns:
            y_encoded: Encoded target vector
        """
        print("Encoding target variable...")
        
        if y.dtype == 'object' or y.dtype.name == 'category':
            y_encoded = self.label_encoder.fit_transform(y)
            print(f"  Classes: {list(self.label_encoder.classes_)}")
            return pd.Series(y_encoded, index=y.index)
        
        return y
    
    def split_data(self, X, y):
        """
        Split data into train and test sets
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"Splitting data (test_size={self.test_size})...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y if len(np.unique(y)) > 1 else None
        )
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test
    
    def preprocess(self, filepath, target_column):
        """
        Complete preprocessing pipeline
        
        Args:
            filepath: Path to CSV file
            target_column: Name of target column
            
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Load data
        X, y = self.load_data(filepath, target_column)
        
        # Handle missing values
        X = self.handle_missing_values(X)
        
        # Encode categorical features
        X = self.encode_categorical_features(X)
        
        # Encode target
        y = self.encode_target(y)
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        print("\nPreprocessing completed successfully!")
        print("="*60)
        
        return X_train, X_test, y_train, y_test, self.feature_names


def validate_data(X, y):
    """
    Validate data for common issues
    
    Args:
        X: Feature matrix
        y: Target vector
    """
    print("\nValidating data...")
    
    # Check for infinite values
    if np.isinf(X.values).any():
        print("  WARNING: Found infinite values in features")
    
    # Check for constant features
    constant_features = X.columns[X.nunique() == 1].tolist()
    if constant_features:
        print(f"  WARNING: Found {len(constant_features)} constant features")
    
    # Check class balance
    class_counts = pd.Series(y).value_counts()
    print(f"  Class distribution: {dict(class_counts)}")
    
    if len(class_counts) > 1:
        imbalance_ratio = class_counts.max() / class_counts.min()
        if imbalance_ratio > 10:
            print(f"  WARNING: Class imbalance ratio: {imbalance_ratio:.2f}")
    
    print("Validation completed!")


if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor(test_size=0.2, random_state=42)
    
    # Replace with your actual file path and target column
    # X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess(
    #     filepath='your_data.csv',
    #     target_column='target'
    # )
    
    print("\nData preprocessing module ready!")
    print("Usage:")
    print("  preprocessor = DataPreprocessor()")
    print("  X_train, X_test, y_train, y_test, features = preprocessor.preprocess(file, target)")