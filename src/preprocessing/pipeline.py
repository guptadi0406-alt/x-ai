"""
Preprocessing pipeline for the Airline Passenger Satisfaction dataset.

Implements reproducible preprocessing that:
- Fits only on training data (no target leakage)
- Handles categorical encoding and numerical scaling
- Preserves feature names for SHAP/LIME compatibility
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Optional


class PreprocessingPipeline:
    """
    Reproducible preprocessing pipeline for the airline satisfaction dataset.
    
    The pipeline:
    1. Identifies and encodes categorical features
    2. Handles missing values
    3. Scales numerical features
    4. Splits into train/test with stratification
    
    All transformations are fitted on training data only to prevent data leakage.
    """
    
    def __init__(self, test_size: float = 0.2, random_seed: int = 42):
        """
        Parameters
        ----------
        test_size : float
            Fraction of data for testing (paper uses 0.2).
        random_seed : int
            Random seed for reproducible splitting.
        """
        self.test_size = test_size
        self.random_seed = random_seed
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: list = []
        self.target_encoder: Optional[LabelEncoder] = None
        self.is_fitted = False
        
    def prepare_features(self, df: pd.DataFrame, feature_columns: list, 
                         target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target from the dataset.
        
        Parameters
        ----------
        df : pd.DataFrame
            The dataset.
        feature_columns : list
            List of feature column names.
        target_column : str
            Name of the target column.
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.Series]
            Features (X) and target (y).
        """
        X = df[feature_columns].copy()
        y = df[target_column].copy()
        
        # Handle missing values before encoding
        for col in X.columns:
            if X[col].isnull().any():
                if X[col].dtype in ['object', 'category']:
                    X[col] = X[col].fillna('Unknown')
                else:
                    X[col] = X[col].fillna(X[col].median())
        
        # Handle missing target values
        if y.isnull().any():
            mask = ~y.isnull()
            X = X[mask]
            y = y[mask]
            print(f"Removed {(~mask).sum()} rows with missing target values")
        
        return X, y
    
    def fit_transform_split(self, X: pd.DataFrame, y: pd.Series
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        """
        Split data, fit preprocessing on training data, and transform both splits.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target vector.
            
        Returns
        -------
        Tuple containing:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Encode target
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)
        
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, 
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y_encoded
        )
        
        print(f"Train/Test split: {len(X_train)}/{len(X_test)} "
              f"(test_size={self.test_size}, seed={self.random_seed})")
        print(f"Train class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")
        
        # Identify categorical and numerical columns
        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        
        # Encode categorical features (fit on train only)
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        for col in cat_cols:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            # Handle unseen categories in test
            test_values = X_test[col].astype(str)
            known_classes = set(le.classes_)
            X_test_encoded[col] = test_values.map(
                lambda x: le.transform([x])[0] if x in known_classes else -1
            )
            self.label_encoders[col] = le
        
        # Store feature names before any further transformation
        self.feature_names = list(X_train_encoded.columns)
        
        # Scale numerical features (fit on train only)
        self.scaler = StandardScaler()
        if num_cols:
            X_train_encoded[num_cols] = self.scaler.fit_transform(X_train_encoded[num_cols])
            X_test_encoded[num_cols] = self.scaler.transform(X_test_encoded[num_cols])
        
        self.is_fitted = True
        
        # Convert to DataFrames with feature names for SHAP/LIME compatibility
        X_train_final = pd.DataFrame(
            X_train_encoded.values, columns=self.feature_names
        )
        X_test_final = pd.DataFrame(
            X_test_encoded.values, columns=self.feature_names
        )
        
        return X_train_final, X_test_final, y_train, y_test, self.feature_names
    
    def get_preprocessing_info(self) -> dict:
        """Return information about the preprocessing pipeline."""
        return {
            'test_size': self.test_size,
            'random_seed': self.random_seed,
            'n_categorical_encoded': len(self.label_encoders),
            'categorical_columns': list(self.label_encoders.keys()),
            'feature_names': self.feature_names,
            'target_classes': list(self.target_encoder.classes_) if self.target_encoder else None,
            'scaler_type': 'StandardScaler',
            'is_fitted': self.is_fitted,
        }
