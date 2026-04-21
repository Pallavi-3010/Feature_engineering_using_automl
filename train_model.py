"""
BigFeat: Scalable Feature Engineering and Interpretable AutoML
Implementation of the BigFeat-vanilla and BigFeat-AutoML framework
"""

import numpy as np
import pandas as pd
import warnings
from itertools import combinations
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score, classification_report

from data_preprocessing import DataPreprocessor, validate_data

warnings.filterwarnings('ignore')


class BigFeatFE:
    """
    BigFeat Feature Engineering Module
    Implements dynamic feature generation and selection
    """
    
    def __init__(self, 
                 n_iterations=7,
                 K=10,
                 alpha=5,
                 eta=0.95,  # CHANGED: from 0.8 to 0.95 (less aggressive redundancy removal)
                 random_state=42):
        """
        Initialize BigFeat-FE
        
        Args:
            n_iterations: Number of feature generation iterations
            K: Batch size multiplier for feature generation (generates K*N features)
            alpha: Number of tree models for stability selection
            eta: Pearson correlation threshold for redundancy removal
            random_state: Random seed
        """
        self.n_iterations = n_iterations
        self.K = K
        self.alpha = alpha
        self.eta = eta
        self.random_state = random_state
        
        # Operators portfolio (unary and binary)
        self.unary_operators = ['square', 'abs', 'sqrt', 'log']
        self.binary_operators = ['add', 'subtract', 'multiply', 'divide']
        
        # Importance scores
        self.feature_importance = None
        self.operator_importance = None
        
        # Feature combination matrix
        self.combination_matrix = None
        
        # Generated features storage
        self.generated_features = []
        self.selected_features = []
        
    def initialize_importance_scores(self, X_train, y_train):
        """
        Initialize feature and operator importance scores
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("Initializing importance scores...")
        
        # Feature importance using Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        rf.fit(X_train, y_train)
        
        self.feature_importance = rf.feature_importances_
        self.feature_importance = self.feature_importance / self.feature_importance.sum()
        
        # Uniform operator importance initially
        n_operators = len(self.unary_operators) + len(self.binary_operators)
        self.operator_importance = np.ones(n_operators) / n_operators
        
        print(f"  Feature importance initialized for {len(self.feature_importance)} features")
        print(f"  Operator importance initialized for {n_operators} operators")
        
    def mine_feature_combinations(self, X_train, y_train):
        """
        Mine feature combinations using tree-based models
        
        Args:
            X_train: Training features
            y_train: Training target
        """
        print("Mining feature combinations...")
        
        n_features = X_train.shape[1]
        self.combination_matrix = np.ones((n_features, n_features))
        
        # Train tree model to find feature interactions
        rf = RandomForestClassifier(
            n_estimators=50, 
            max_depth=10,
            random_state=self.random_state
        )
        rf.fit(X_train, y_train)
        
        # Extract feature pairs from tree paths
        for tree in rf.estimators_:
            tree_structure = tree.tree_
            features_in_tree = tree_structure.feature
            
            # Get unique features used in tree
            used_features = [f for f in features_in_tree if f != -2]  # -2 is leaf
            
            # Increment combination matrix for feature pairs
            for i, j in combinations(used_features, 2):
                if i < n_features and j < n_features:
                    self.combination_matrix[i, j] += 1
                    self.combination_matrix[j, i] += 1
        
        print(f"  Combination matrix computed: {self.combination_matrix.shape}")
        
    def apply_unary_operator(self, feature, operator):
        """
        Apply unary operator to a feature
        
        Args:
            feature: Input feature array
            operator: Operator name
            
        Returns:
            Transformed feature
        """
        try:
            if operator == 'square':
                return np.square(feature)
            elif operator == 'abs':
                return np.abs(feature)
            elif operator == 'sqrt':
                return np.sqrt(np.abs(feature))  # Handle negative values
            elif operator == 'log':
                return np.log(np.abs(feature) + 1e-10)  # Avoid log(0)
            else:
                return feature
        except:
            return feature
    
    def apply_binary_operator(self, feature1, feature2, operator):
        """
        Apply binary operator to two features
        
        Args:
            feature1: First input feature
            feature2: Second input feature
            operator: Operator name
            
        Returns:
            Combined feature
        """
        try:
            if operator == 'add':
                return feature1 + feature2
            elif operator == 'subtract':
                return feature1 - feature2
            elif operator == 'multiply':
                return feature1 * feature2
            elif operator == 'divide':
                return feature1 / (feature2 + 1e-10)  # Avoid division by zero
            else:
                return feature1
        except:
            return feature1
    
    def generate_features(self, X_train, iteration):
        """
        Generate K*N new features dynamically
        
        Args:
            X_train: Training features (pandas DataFrame)
            iteration: Current iteration number
            
        Returns:
            X_new: DataFrame with newly generated features
        """
        print(f"\nIteration {iteration + 1}/{self.n_iterations}")
        print(f"  Generating {self.K * X_train.shape[1]} new features...")
        
        n_features = X_train.shape[1]
        n_new_features = self.K * n_features
        
        new_features = []
        feature_descriptions = []
        
        # Sample features based on importance
        feature_probs = self.feature_importance / self.feature_importance.sum()
        
        for i in range(n_new_features):
            # Randomly decide: unary or binary operation
            if np.random.random() < 0.4:  # 40% unary operations
                # Unary operation
                feat_idx = np.random.choice(n_features, p=feature_probs)
                operator = np.random.choice(self.unary_operators)
                
                new_feat = self.apply_unary_operator(
                    X_train.iloc[:, feat_idx].values, 
                    operator
                )
                
                desc = f"{operator}({X_train.columns[feat_idx]})"
                
            else:  # 60% binary operations
                # Binary operation
                feat_idx1 = np.random.choice(n_features, p=feature_probs)
                
                # FIXED: Safe combination matrix access
                matrix_size = self.combination_matrix.shape[0]
                
                if feat_idx1 < matrix_size and n_features <= matrix_size:
                    # Use combination matrix
                    comb_probs = self.combination_matrix[feat_idx1, :n_features].copy()
                    comb_probs[feat_idx1] = 0
                    
                    if comb_probs.sum() > 0:
                        comb_probs = comb_probs / comb_probs.sum()
                    else:
                        # Fallback to uniform
                        comb_probs = np.ones(n_features)
                        comb_probs[feat_idx1] = 0
                        comb_probs = comb_probs / comb_probs.sum()
                else:
                    # Use uniform distribution as fallback
                    comb_probs = np.ones(n_features)
                    comb_probs[feat_idx1] = 0
                    comb_probs = comb_probs / comb_probs.sum()
                
                feat_idx2 = np.random.choice(n_features, p=comb_probs)
                operator = np.random.choice(self.binary_operators)
                
                new_feat = self.apply_binary_operator(
                    X_train.iloc[:, feat_idx1].values,
                    X_train.iloc[:, feat_idx2].values,
                    operator
                )
                
                desc = f"{X_train.columns[feat_idx1]}_{operator}_{X_train.columns[feat_idx2]}"
            
            # Handle invalid values
            new_feat = np.nan_to_num(new_feat, nan=0.0, posinf=0.0, neginf=0.0)
            
            new_features.append(new_feat)
            feature_descriptions.append(desc)
        
        X_new = pd.DataFrame(
            np.column_stack(new_features),
            columns=feature_descriptions,
            index=X_train.index
        )
        
        print(f"  Generated {X_new.shape[1]} features")
        
        return X_new
    
    def stability_feature_selection(self, X_combined, y_train, n_select):
        """
        Select top N features using stability selection
        
        Args:
            X_combined: Combined original and generated features
            y_train: Training target
            n_select: Number of features to select
            
        Returns:
            selected_indices: Indices of selected features
        """
        print(f"  Selecting top {n_select} features using stability selection...")
        
        n_features = X_combined.shape[1]
        feature_scores = np.zeros(n_features)
        
        # Train alpha tree models on random subsets
        for i in range(self.alpha):
            # Random subset of features and samples
            n_subset_features = int(0.8 * n_features)
            n_subset_samples = int(0.8 * len(y_train))
            
            feature_indices = np.random.choice(
                n_features, 
                size=n_subset_features, 
                replace=False
            )
            sample_indices = np.random.choice(
                len(y_train), 
                size=n_subset_samples, 
                replace=False
            )
            
            X_subset = X_combined.iloc[sample_indices, feature_indices]
            y_subset = y_train.iloc[sample_indices]
            
            # Clean data: replace inf/nan values aggressively
            X_subset_clean = X_subset.values.copy()
            X_subset_clean = np.nan_to_num(X_subset_clean, nan=0.0, posinf=0.0, neginf=0.0)
            # Clip extremely large values that might overflow float32
            X_subset_clean = np.clip(X_subset_clean, -1e10, 1e10)
            
            # Train Random Forest
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=self.random_state + i
            )
            rf.fit(X_subset_clean, y_subset)
            
            # Aggregate importance scores
            feature_scores[feature_indices] += rf.feature_importances_
        
        # Normalize scores
        feature_scores = feature_scores / self.alpha
        
        # Sort by importance
        sorted_indices = np.argsort(feature_scores)[::-1]
        
        return sorted_indices[:n_select]
    
    def remove_redundant_features(self, X_combined, selected_indices):
        """
        Remove redundant features based on Pearson correlation
        
        Args:
            X_combined: Combined feature matrix
            selected_indices: Initially selected feature indices
            
        Returns:
            final_indices: Final selected feature indices after redundancy removal
        """
        print(f"  Removing redundant features (threshold={self.eta})...")
        
        final_indices = []
        
        for idx in selected_indices:
            is_redundant = False
            
            for final_idx in final_indices:
                try:
                    corr, _ = pearsonr(
                        X_combined.iloc[:, idx], 
                        X_combined.iloc[:, final_idx]
                    )
                    
                    if abs(corr) > self.eta:
                        is_redundant = True
                        break
                except:
                    continue
            
            if not is_redundant:
                final_indices.append(idx)
            
            if len(final_indices) >= len(selected_indices):
                break
        
        # FIXED: Ensure minimum number of features
        min_features = max(50, len(selected_indices) // 10)
        
        if len(final_indices) < min_features:
            print(f"  WARNING: Only {len(final_indices)} features after redundancy removal")
            print(f"  Adding back features to reach minimum of {min_features}")
            
            # Add back some features that were marked redundant
            for idx in selected_indices:
                if idx not in final_indices:
                    final_indices.append(idx)
                if len(final_indices) >= min_features:
                    break
        
        removed = len(selected_indices) - len(final_indices)
        print(f"  Removed {removed} redundant features")
        
        return final_indices
    
    def update_operator_importance(self, selected_feature_names):
        """
        Update operator importance based on selected features
        
        Args:
            selected_feature_names: Names of selected features
        """
        operator_counts = np.zeros(len(self.unary_operators) + len(self.binary_operators))
        
        for feat_name in selected_feature_names:
            # Check which operators appear in feature name
            for i, op in enumerate(self.unary_operators):
                if op in feat_name:
                    operator_counts[i] += 1
            
            for i, op in enumerate(self.binary_operators):
                if op in feat_name:
                    operator_counts[len(self.unary_operators) + i] += 1
        
        # Update importance (incremental average)
        if operator_counts.sum() > 0:
            self.operator_importance = (self.operator_importance + operator_counts) / 2
            self.operator_importance = self.operator_importance / self.operator_importance.sum()
    
    def fit_transform(self, X_train, y_train):
        """
        Complete feature engineering pipeline
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            X_transformed: Engineered features
        """
        print("\n" + "="*60)
        print("BigFeat Feature Engineering Pipeline")
        print("="*60)
        
        # Initialize
        self.initialize_importance_scores(X_train, y_train)
        self.mine_feature_combinations(X_train, y_train)
        
        X_current = X_train.copy()
        n_original_features = X_train.shape[1]
        
        # Iterative feature generation and selection
        for iteration in range(self.n_iterations):
            # Generate new features
            X_new = self.generate_features(X_current, iteration)
            
            # Combine with current features
            X_combined = pd.concat([X_current, X_new], axis=1)
            
            print(f"  Total features: {X_combined.shape[1]}")
            
            # Feature selection
            selected_indices = self.stability_feature_selection(
                X_combined, 
                y_train, 
                n_original_features
            )
            
            # Remove redundant features
            final_indices = self.remove_redundant_features(
                X_combined, 
                selected_indices
            )
            
            # Update current feature set
            X_current = X_combined.iloc[:, final_indices]
            
            print(f"  Final features after selection: {X_current.shape[1]}")
            
            # Update importance scores
            self.feature_importance = np.ones(X_current.shape[1]) / X_current.shape[1]
            self.update_operator_importance(X_current.columns)
        
        self.selected_features = X_current.columns.tolist()
        
        print("\n" + "="*60)
        print(f"Feature Engineering Completed!")
        print(f"Final feature count: {X_current.shape[1]}")
        print("="*60 + "\n")
        
        return X_current
    
    def transform(self, X):
        """
        Transform new data using selected features
        Note: This is a simplified version
        
        Args:
            X: Input features
            
        Returns:
            X_transformed: Transformed features
        """
        # In a full implementation, we would need to store the exact
        # transformation pipeline and apply it here
        # For now, we return the input
        return X


class BigFeatAutoML:
    """
    BigFeat AutoML Module
    Performs feature engineering, model selection, and hyperparameter tuning
    """
    
    def __init__(self, 
                 time_budget=600,  # 10 minutes
                 n_iterations_fe=5,
                 random_state=42):
        """
        Initialize BigFeat-AutoML
        
        Args:
            time_budget: Time budget in seconds
            n_iterations_fe: Number of feature engineering iterations
            random_state: Random seed
        """
        self.time_budget = time_budget
        self.n_iterations_fe = n_iterations_fe
        self.random_state = random_state
        
        # Feature engineering module
        self.feature_engineer = BigFeatFE(
            n_iterations=n_iterations_fe,
            random_state=random_state
        )
        
        # Model search space (interpretable models only)
        self.model_space = self._define_model_space()
        
        # Best model and score
        self.best_model = None
        self.best_params = None
        self.best_score = -np.inf
        
    def _define_model_space(self):
        """
        Define search space for interpretable models
        
        Returns:
            model_space: Dictionary of models and hyperparameters
        """
        model_space = {
            'RandomForest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [10, 50, 100, 250],
                    'max_depth': [None, 5, 10, 20],
                    'class_weight': [None, 'balanced']
                }
            },
            'LogisticRegression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear'],
                    'class_weight': [None, 'balanced'],
                    'max_iter': [1000]
                }
            },
            'DecisionTree': {
                'model': DecisionTreeClassifier,
                'params': {
                    'max_depth': [5, 10, 25, None],
                    'min_samples_split': [2, 5, 10],
                    'class_weight': [None, 'balanced']
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [10, 50, 100, 250],
                    'max_depth': [5, 10, 20]
                }
            }
        }
        
        return model_space
    
    def random_search(self, X_train, y_train, n_trials=50):
        """
        Random search for model selection and hyperparameter tuning
        
        Args:
            X_train: Training features
            y_train: Training target
            n_trials: Number of random trials
            
        Returns:
            best_model: Best model found
            best_params: Best hyperparameters
        """
        print("\n" + "="*60)
        print("BigFeat AutoML: Model Selection & Hyperparameter Tuning")
        print("="*60)
        print(f"Running {n_trials} random search trials...\n")
        
        for trial in range(n_trials):
            # Randomly select a model
            model_name = np.random.choice(list(self.model_space.keys()))
            model_config = self.model_space[model_name]
            
            # Randomly select hyperparameters
            params = {}
            for param_name, param_values in model_config['params'].items():
                params[param_name] = np.random.choice(param_values)
            
            # Handle solver-penalty compatibility for LogisticRegression
            if model_name == 'LogisticRegression':
                if params['penalty'] == 'l1':
                    params['solver'] = 'liblinear'
            
            try:
                # Create model
                model = model_config['model'](**params, random_state=self.random_state)
                
                # Cross-validation score
                scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=3, 
                    scoring='f1_weighted',
                    n_jobs=-1
                )
                score = scores.mean()
                
                # Update best model
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = model
                    self.best_params = {'model': model_name, **params}
                    
                    print(f"Trial {trial + 1}/{n_trials}: New best model!")
                    print(f"  Model: {model_name}")
                    print(f"  Params: {params}")
                    print(f"  CV F1 Score: {score:.4f}")
                
            except Exception as e:
                # Skip invalid configurations
                continue
        
        print("\n" + "="*60)
        print("Random Search Completed!")
        print(f"Best Model: {self.best_params['model']}")
        print(f"Best CV F1 Score: {self.best_score:.4f}")
        print("="*60 + "\n")
        
        return self.best_model, self.best_params
    
    def fit(self, X_train, y_train):
        """
        Complete AutoML pipeline: Feature Engineering + Model Selection
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            self
        """
        print("\n" + "="*60)
        print("BigFeat AutoML Pipeline")
        print("="*60)
        
        # Step 1: Feature Engineering
        print("\nStep 1: Feature Engineering")
        X_engineered = self.feature_engineer.fit_transform(X_train, y_train)
        
        # Step 2: Model Selection and Hyperparameter Tuning
        print("\nStep 2: Model Selection")
        self.best_model, self.best_params = self.random_search(
            X_engineered, y_train, n_trials=50
        )
        
        # Step 3: Train final model on all data
        print("\nStep 3: Training Final Model")
        self.best_model.fit(X_engineered, y_train)
        print("Final model training completed!")
        
        return self
    
    def predict(self, X_test):
        """
        Make predictions on test data
        
        Args:
            X_test: Test features
            
        Returns:
            predictions: Predicted labels
        """
        # Note: In full implementation, we would apply the same
        # feature engineering transformations to X_test
        # For simplicity, we're using X_test directly
        return self.best_model.predict(X_test)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            metrics: Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        print("\n" + "="*60)
        print("Test Set Evaluation")
        print("="*60)
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        print("="*60 + "\n")
        
        return metrics


if __name__ == "__main__":
    print("BigFeat: Feature Engineering and AutoML Framework")
    print("Import this module to use BigFeatFE and BigFeatAutoML classes")