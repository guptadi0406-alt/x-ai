"""
SHAP explainer module.

Implements SHAP explanations for all four classifiers, generating both
global and local explanations as described in the paper.

The paper uses SHAP to:
1. Generate global feature importance rankings per model (Figure 2, Table 2)
2. Compare rankings across models to demonstrate model-dependency
"""

import logging
import numpy as np
import pandas as pd
import shap
from typing import Dict, Any, Optional
import warnings

logger = logging.getLogger(__name__)


def get_shap_explainer(model, model_type: str, X_train: pd.DataFrame,
                       n_background: int = 100):
    """
    Create the appropriate SHAP explainer for each model type.

    Parameters
    ----------
    model : trained classifier
        The model to explain.
    model_type : str
        Model abbreviation: 'DT', 'LR', 'LGBM', or 'SVC'.
    X_train : pd.DataFrame
        Training data for background samples.
    n_background : int
        Number of background samples for KernelSHAP.

    Returns
    -------
    shap.Explainer
        The SHAP explainer instance.
    """
    if model_type in ('LGBM', 'DT'):
        # TreeExplainer is exact for tree models
        return shap.TreeExplainer(model)
    elif model_type == 'LR':
        # LinearExplainer for linear models
        masker = shap.maskers.Independent(data=X_train)
        return shap.LinearExplainer(model, masker=masker)
    elif model_type == 'SVC':
        # KernelSHAP for model-agnostic explanation
        # Note: KernelSHAP is a stochastic approximation. Reproducibility
        # is controlled by the global numpy seed set in run_all.py.
        background = shap.sample(X_train, min(n_background, len(X_train)))
        return shap.KernelExplainer(model.predict_proba, background)
    else:
        # Fallback to KernelSHAP
        logger.warning("Unknown model_type '%s'; falling back to KernelSHAP.", model_type)
        background = shap.sample(X_train, min(n_background, len(X_train)))
        return shap.KernelExplainer(model.predict, background)


def compute_shap_values(explainer, X_test: pd.DataFrame, model_type: str) -> np.ndarray:
    """
    Compute SHAP values for test data.

    Parameters
    ----------
    explainer : shap.Explainer
        The SHAP explainer.
    X_test : pd.DataFrame
        Test data.
    model_type : str
        Model abbreviation.

    Returns
    -------
    np.ndarray
        SHAP values matrix (n_samples, n_features).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap_values = explainer.shap_values(X_test)

        # Handle Explanation objects (newer SHAP versions)
        if hasattr(shap_values, 'values'):
            shap_values = shap_values.values

        # Handle list output (binary classification returns [class0, class1])
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class

        result = np.asarray(shap_values)

        # Handle 3D arrays (n_samples, n_features, n_classes)
        if result.ndim == 3:
            result = result[:, :, 1]  # Use positive class

        # Ensure 2D
        if result.ndim == 1:
            result = result.reshape(1, -1)

    if result.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"SHAP values width ({result.shape[1]}) does not match "
            f"number of features ({X_test.shape[1]}) for model '{model_type}'."
        )

    return result


def get_global_feature_importance(shap_values: np.ndarray,
                                  feature_names: list) -> pd.DataFrame:
    """
    Calculate global feature importance from SHAP values.

    Global importance = mean absolute SHAP value across all test instances.

    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix.
    feature_names : list
        Feature names.

    Returns
    -------
    pd.DataFrame
        Features ranked by importance (descending), with Rank column.
    """
    if shap_values.shape[1] != len(feature_names):
        raise ValueError(
            f"shap_values has {shap_values.shape[1]} columns but "
            f"{len(feature_names)} feature names were provided."
        )

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).flatten().tolist()

    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean |SHAP|': mean_abs_shap,
    })

    # Sort once, then assign Rank (no dead pre-sort rank column)
    importance_df = importance_df.sort_values('Mean |SHAP|', ascending=False)
    importance_df['Rank'] = range(1, len(feature_names) + 1)
    importance_df = importance_df.reset_index(drop=True)

    return importance_df


def compute_all_shap(model_results: Dict[str, Dict], X_train: pd.DataFrame,
                     X_test: pd.DataFrame, feature_names: list,
                     n_background: int = 100) -> Dict[str, Dict]:
    """
    Compute SHAP explanations for all trained models.

    Parameters
    ----------
    model_results : dict
        Results from train_and_evaluate_all().
    X_train : pd.DataFrame
        Training data.
    X_test : pd.DataFrame
        Test data.
    feature_names : list
        Feature names.
    n_background : int
        Background samples for KernelSHAP.

    Returns
    -------
    dict
        SHAP results for each model (shap_values, feature_importance, feature_ranking).
        Note: explainer objects are intentionally excluded from the returned dict
        because they are not JSON-serializable.
    """
    shap_results: Dict[str, Dict] = {}

    for model_type, res in model_results.items():
        print(f"\nComputing SHAP for {res['name']} ({model_type})...")
        if model_type == 'SVC':
            print("  (KernelSHAP — this may take a few minutes)")

        model = res['model']

        explainer = get_shap_explainer(model, model_type, X_train, n_background)
        shap_values = compute_shap_values(explainer, X_test, model_type)
        importance = get_global_feature_importance(shap_values, feature_names)

        # Intentionally omit 'explainer' key: explainer objects are not
        # JSON-serializable and should not be stored in the results dict.
        shap_results[model_type] = {
            'shap_values': shap_values,
            'feature_importance': importance,
            'feature_ranking': importance['Feature'].tolist(),
        }

        print(f"  Top 5 features: {importance['Feature'].head(5).tolist()}")
        print(f"  SHAP values shape: {shap_values.shape}")

    return shap_results


def compare_rankings(shap_results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table of feature rankings across models.
    Reproduces the structure of Table 2 in the paper.

    Parameters
    ----------
    shap_results : dict
        SHAP results for each model.

    Returns
    -------
    pd.DataFrame
        Ranking comparison table (Rank × Model).
    """
    ranking_data = {
        model_type: res['feature_ranking']
        for model_type, res in shap_results.items()
    }

    max_features = max(len(v) for v in ranking_data.values())

    comparison = pd.DataFrame({
        model: ranking + [''] * (max_features - len(ranking))
        for model, ranking in ranking_data.items()
    })
    comparison.index = range(1, len(comparison) + 1)
    comparison.index.name = 'Rank'

    return comparison
