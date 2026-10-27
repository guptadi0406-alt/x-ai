"""
Preprocessing pipeline for the Airline Passenger Satisfaction dataset.

Implements reproducible preprocessing that:
- Fits only on training data (no target leakage)
- Handles categorical encoding and numerical scaling
- Preserves feature names for SHAP/LIME compatibility

Note on encoder choice:
    LabelEncoder is used for categorical features for SHAP/LIME compatibility.
    This encodes categories as integers (e.g., Female=0, Male=1). While this
    implies ordinal ordering, tree-based and linear models handle these integer
    codes well, and SHAP/LIME work at the encoded feature level. For a fully
    nominal-safe implementation, OneHotEncoder would be preferred but would
    change the feature space and require post-hoc name mapping.

Documented deviations from sklearn defaults (required for this pipeline):
    - LR: max_iter=1000 (default 100 is insufficient for convergence)
    - SVC: probability=True (required for LIME's predict_proba interface)
"""

import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


class PreprocessingPipeline:
    """
    Reproducible preprocessing pipeline for the airline satisfaction dataset.

    The pipeline:
    1. Identifies and encodes categorical features (fit on train only)
    2. Handles numeric missing values using *training-set* median (no leakage)
    3. Scales numerical features with StandardScaler (fit on train only)
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
        # Stores per-column train medians used for imputation (for inspection)
        self._train_medians: Dict[str, float] = {}

    def prepare_features(self, df: pd.DataFrame, feature_columns: list,
                         target_column: str) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extract features and target from the dataset.

        Numeric missing-value imputation is deferred to fit_transform_split()
        so that medians are computed from training data only (no leakage).

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

        # Fill missing categorical values with 'Unknown' here — safe to do
        # before the split because it is a fixed string, not a learned statistic.
        for col in X.columns:
            if X[col].isnull().any() and X[col].dtype in ['object', 'category']:
                X[col] = X[col].fillna('Unknown')

        # Remove rows with missing target
        if y.isnull().any():
            mask = ~y.isnull()
            n_removed = int((~mask).sum())
            X = X[mask]
            y = y[mask]
            logger.warning("Removed %d rows with missing target values", n_removed)
            print(f"Removed {n_removed} rows with missing target values")

        return X, y

    def fit_transform_split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list]:
        """
        Split data, fit preprocessing on training data, and transform both splits.

        Numeric missing values are imputed using the *training-set* median, which
        is then applied to the test set — this prevents any test-data leakage
        through the imputation step.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (raw, pre-split).
        y : pd.Series
            Target vector.

        Returns
        -------
        Tuple containing:
            X_train (pd.DataFrame), X_test (pd.DataFrame),
            y_train (np.ndarray), y_test (np.ndarray),
            feature_names (list)
        """
        # --- Encode target ---
        self.target_encoder = LabelEncoder()
        y_encoded = self.target_encoder.fit_transform(y)

        # --- Stratified split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=y_encoded,
        )

        print(f"Train/Test split: {len(X_train)}/{len(X_test)} "
              f"(test_size={self.test_size}, seed={self.random_seed})")
        print(f"Train class distribution: {np.bincount(y_train)}")
        print(f"Test class distribution: {np.bincount(y_test)}")

        cat_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X_train.select_dtypes(include=['number']).columns.tolist()

        # --- Numeric missing-value imputation: train median only (no leakage) ---
        X_train = X_train.copy()
        X_test = X_test.copy()
        for col in num_cols:
            if X_train[col].isnull().any() or X_test[col].isnull().any():
                train_median = float(X_train[col].median())
                self._train_medians[col] = train_median
                X_train[col] = X_train[col].fillna(train_median)
                X_test[col] = X_test[col].fillna(train_median)
                logger.debug("Imputed '%s' with train median=%.4f", col, train_median)

        # --- Encode categorical features (fit on train only) ---
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()

        for col in cat_cols:
            le = LabelEncoder()
            X_train_encoded[col] = le.fit_transform(X_train[col].astype(str))
            known_classes = set(le.classes_)
            # Map test values; unseen categories get -1
            X_test_encoded[col] = X_test[col].astype(str).map(
                lambda x, _le=le, _kc=known_classes: (
                    int(_le.transform([x])[0]) if x in _kc else -1
                )
            )
            self.label_encoders[col] = le

        # Store feature names after encoding (before scaling)
        self.feature_names = list(X_train_encoded.columns)

        # --- Scale numerical features (fit on train only) ---
        self.scaler = StandardScaler()
        if num_cols:
            X_train_encoded[num_cols] = self.scaler.fit_transform(
                X_train_encoded[num_cols]
            )
            X_test_encoded[num_cols] = self.scaler.transform(
                X_test_encoded[num_cols]
            )

        self.is_fitted = True

        # Return as DataFrames with feature names (needed for SHAP/LIME)
        X_train_final = pd.DataFrame(
            X_train_encoded.values, columns=self.feature_names
        )
        X_test_final = pd.DataFrame(
            X_test_encoded.values, columns=self.feature_names
        )

        return X_train_final, X_test_final, y_train, y_test, self.feature_names

    def get_preprocessing_info(self) -> dict:
        """Return information about the fitted preprocessing pipeline.

        Raises
        ------
        RuntimeError
            If called before fit_transform_split().
        """
        if not self.is_fitted:
            raise RuntimeError(
                "Pipeline has not been fitted yet. "
                "Call fit_transform_split() first."
            )
        return {
            'test_size': self.test_size,
            'random_seed': self.random_seed,
            'n_categorical_encoded': len(self.label_encoders),
            'categorical_columns': list(self.label_encoders.keys()),
            'feature_names': self.feature_names,
            'target_classes': (
                list(self.target_encoder.classes_) if self.target_encoder else None
            ),
            'scaler_type': 'StandardScaler',
            # Expose fitted scaler mean for leakage verification in tests
            'scaler_mean': (
                self.scaler.mean_.tolist() if self.scaler is not None else None
            ),
            'is_fitted': self.is_fitted,
            'train_medians_used': self._train_medians,
        }
