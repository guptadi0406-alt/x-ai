"""
LIME explainer module.

Implements LIME explanations for all four classifiers, generating
local explanations as described in the paper (Figure 3).

LIME approximates the complex model with a local linear model for
each individual instance, reporting coefficient values as feature weights.

Note on LIME feature names:
    LIME discretizes continuous features into bins (e.g., "Age > 40.00").
    These bin-string keys differ from the raw feature names used by SHAP,
    so LIME and SHAP feature rankings are not directly comparable by name.
    This is documented in Section 10 of the paper.

Note on seed handling:
    LIME randomness is controlled at the LimeTabularExplainer level via
    ``random_state``. The ``random_seed`` parameter in explain_instance()
    is not forwarded to LIME's internal perturbation sampler (LIME does
    not expose a per-call seed). To vary seeds, create a new explainer
    with a different random_state (as done in lime_stability_analysis).
"""

import logging
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from typing import Dict, Any, List, Optional
import warnings

logger = logging.getLogger(__name__)


def create_lime_explainer(X_train: pd.DataFrame, feature_names: list,
                          class_names: list = None,
                          random_seed: int = 42) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Create a LIME tabular explainer.

    The random_seed controls LIME's internal perturbation sampling for
    reproducibility. To vary seeds across runs, call this function again
    with a different seed (see lime_stability_analysis).

    Parameters
    ----------
    X_train : pd.DataFrame
        Training data for computing feature statistics.
    feature_names : list
        Feature names.
    class_names : list, optional
        Class labels.
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    lime.lime_tabular.LimeTabularExplainer
    """
    if class_names is None:
        class_names = ['Non-Satisfied', 'Satisfied']

    return lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train.values if isinstance(X_train, pd.DataFrame) else X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=random_seed,
        discretize_continuous=True,
    )


def explain_instance(lime_explainer, model, instance: np.ndarray,
                     n_features: int = 22, n_samples: int = 5000,
                     random_seed: int = 42) -> dict:
    """
    Generate a LIME explanation for a single instance.

    Note: ``random_seed`` is accepted for API symmetry but LIME does not
    support a per-call seed. Seed control must be done at the
    LimeTabularExplainer level (random_state parameter). Use
    lime_stability_analysis() to run with multiple seeds.

    Parameters
    ----------
    lime_explainer : LimeTabularExplainer
        The LIME explainer (seed set at creation time).
    model : trained classifier
        Must have a predict_proba method.
    instance : np.ndarray
        Single instance to explain.
    n_features : int
        Number of features to include in explanation.
    n_samples : int
        Number of perturbation samples.
    random_seed : int
        Accepted for API consistency; seed is controlled at explainer level.

    Returns
    -------
    dict
        Explanation including weights, predicted probabilities, and feature values.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explanation = lime_explainer.explain_instance(
            data_row=instance,
            predict_fn=model.predict_proba,
            num_features=n_features,
            num_samples=n_samples,
        )

    # Feature weights use LIME's discretized bin strings as keys
    # (e.g., "Online boarding > 3.00") — different from raw SHAP feature names.
    feature_weights = dict(explanation.as_list())
    proba = explanation.predict_proba

    return {
        'explanation': explanation,
        'feature_weights': feature_weights,
        'predicted_proba': proba,
        'intercept': explanation.intercept if hasattr(explanation, 'intercept') else None,
        'local_prediction': explanation.local_pred if hasattr(explanation, 'local_pred') else None,
    }


def compute_lime_for_instance(model_results: Dict[str, Dict],
                              lime_explainer,
                              instance: np.ndarray,
                              instance_idx: int,
                              n_features: int = 22,
                              n_samples: int = 5000) -> Dict[str, Dict]:
    """
    Compute LIME explanations for the same instance across all models.

    This reproduces Figure 3 from the paper, showing how different models
    produce different local explanations for the same observation.

    Parameters
    ----------
    model_results : dict
        Results from train_and_evaluate_all().
    lime_explainer : LimeTabularExplainer
        LIME explainer.
    instance : np.ndarray
        The instance to explain.
    instance_idx : int
        Index for reference logging.
    n_features : int
        Number of features.
    n_samples : int
        Perturbation samples.

    Returns
    -------
    dict
        LIME results for each model.
    """
    lime_results: Dict[str, Dict] = {}

    for model_type, res in model_results.items():
        print(f"Computing LIME for {res['name']} ({model_type}), "
              f"instance {instance_idx}...")

        exp = explain_instance(
            lime_explainer, res['model'], instance,
            n_features=n_features, n_samples=n_samples,
        )
        lime_results[model_type] = exp

        print(f"  Predicted proba: {exp['predicted_proba']}")
        print(f"  Top 3 features: {list(exp['feature_weights'].keys())[:3]}")

    return lime_results


def lime_stability_analysis(lime_explainer, model, instance: np.ndarray,
                            X_train=None, feature_names: list = None,
                            class_names: list = None,
                            n_features: int = 22, n_samples: int = 5000,
                            seeds: list = None) -> Dict[str, Any]:
    """
    Assess LIME explanation stability by running with different random seeds.

    LIME uses random perturbations; explanations can vary across runs.
    This experiment quantifies that instability by creating a fresh
    LimeTabularExplainer for each seed (the only way to vary LIME's seed).

    Parameters
    ----------
    lime_explainer : LimeTabularExplainer
        Used only to extract training_data/feature_names if not provided.
    model : trained classifier
    instance : np.ndarray
    X_train : array-like, optional
        Training data for creating new explainers.
    feature_names : list, optional
    class_names : list, optional
    n_features : int
    n_samples : int
    seeds : list
        Random seeds to test.

    Returns
    -------
    dict
        Stability analysis results including per-seed weights and rankings.
    """
    if seeds is None:
        seeds = list(range(10))

    # Resolve training data from explicit arg or explainer internals
    if X_train is None:
        for attr in ['training_data', '_training_data', 'train']:
            if hasattr(lime_explainer, attr):
                X_train = getattr(lime_explainer, attr)
                break
    if X_train is None:
        raise ValueError("X_train must be provided for LIME stability analysis")

    train_array = X_train.values if hasattr(X_train, 'values') else X_train

    if feature_names is None:
        feature_names = getattr(lime_explainer, 'feature_names', None)
    if class_names is None:
        class_names = getattr(lime_explainer, 'class_names', None)

    all_weights: List[dict] = []
    all_rankings: List[List[str]] = []

    for seed in seeds:
        # Seed must be set at explainer creation — not per-call
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            training_data=train_array,
            feature_names=feature_names,
            class_names=class_names,
            mode='classification',
            random_state=seed,
            discretize_continuous=True,
        )

        exp = explain_instance(lime_exp, model, instance, n_features, n_samples, seed)
        all_weights.append(exp['feature_weights'])
        all_rankings.append(list(exp['feature_weights'].keys()))

    return {
        'seeds': seeds,
        'all_weights': all_weights,
        'all_rankings': all_rankings,
        'n_runs': len(seeds),
        'note': (
            'Rankings use LIME discretized bin strings (e.g., "Age > 40"), '
            'not raw feature names. Direct comparison with SHAP rankings '
            'requires name normalization.'
        ),
    }


def lime_stability_all_models(model_results: Dict[str, Dict],
                              X_train, feature_names: list,
                              class_names: list,
                              instance: np.ndarray,
                              n_features: int = 22,
                              n_samples: int = 5000,
                              seeds: list = None) -> Dict[str, Dict]:
    """
    Run LIME stability analysis across ALL four models for a single instance.

    This extends the single-model stability analysis in the paper to compare
    LIME stability across model types, supporting the paper's claim about
    model-dependent instability.

    Parameters
    ----------
    model_results : dict
        Results from train_and_evaluate_all().
    X_train : pd.DataFrame or np.ndarray
    feature_names : list
    class_names : list
    instance : np.ndarray
        Instance to explain.
    n_features : int
    n_samples : int
    seeds : list

    Returns
    -------
    dict
        Per-model stability results.
    """
    if seeds is None:
        seeds = list(range(10))

    results: Dict[str, Dict] = {}
    for model_type, res in model_results.items():
        print(f"LIME stability analysis: {res['name']} ({model_type})...")
        # Create a baseline explainer just to satisfy the function signature
        baseline_exp = create_lime_explainer(
            X_train if hasattr(X_train, 'values') else pd.DataFrame(X_train),
            feature_names, class_names, random_seed=seeds[0],
        )
        stability = lime_stability_analysis(
            baseline_exp, res['model'], instance,
            X_train=X_train, feature_names=feature_names,
            class_names=class_names,
            n_features=n_features, n_samples=n_samples, seeds=seeds,
        )
        results[model_type] = stability
        print(f"  Completed {stability['n_runs']} runs for {model_type}")

    return results
