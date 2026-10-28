"""
LIME explainer module.

Implements LIME explanations for all four classifiers, generating
local explanations as described in the paper (Figure 3).

LIME approximates the complex model with a local linear model for
each individual instance, reporting coefficient values as feature weights.
"""

import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
from typing import Dict, Any, List, Optional
import warnings


def create_lime_explainer(X_train: pd.DataFrame, feature_names: list,
                          class_names: list = None,
                          random_seed: int = 42) -> lime.lime_tabular.LimeTabularExplainer:
    """
    Create a LIME tabular explainer.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training data for computing statistics.
    feature_names : list
        Feature names.
    class_names : list, optional
        Class labels.
    random_seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    lime.lime_tabular.LimeTabularExplainer
        The LIME explainer.
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
    
    Parameters
    ----------
    lime_explainer : LimeTabularExplainer
        The LIME explainer.
    model : trained classifier
        Must have a predict_proba method.
    instance : np.ndarray
        Single instance to explain.
    n_features : int
        Number of features to include in explanation.
    n_samples : int
        Number of perturbation samples.
    random_seed : int
        Random seed.
        
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
    
    # Extract feature weights
    feature_weights = {}
    for feat, weight in explanation.as_list():
        feature_weights[feat] = weight
    
    # Get predicted probabilities
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
        Index for reference.
    n_features : int
        Number of features.
    n_samples : int
        Perturbation samples.
        
    Returns
    -------
    dict
        LIME results for each model.
    """
    lime_results = {}
    
    for model_type, res in model_results.items():
        print(f"Computing LIME for {res['name']} ({model_type}), instance {instance_idx}...")
        
        model = res['model']
        
        exp = explain_instance(
            lime_explainer, model, instance, 
            n_features=n_features, n_samples=n_samples
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
    Assess LIME explanation stability by repeating with different random seeds.
    
    LIME uses random perturbations, so explanations can vary across runs.
    This experiment quantifies that instability.
    
    Parameters
    ----------
    lime_explainer : LimeTabularExplainer
        LIME explainer (used as fallback for training data).
    model : trained classifier
        The model to explain.
    instance : np.ndarray
        Instance to explain.
    X_train : array-like, optional
        Training data for creating new explainers.
    feature_names : list, optional
        Feature names.
    class_names : list, optional
        Class names.
    n_features : int
        Number of features.
    n_samples : int
        Perturbation samples.
    seeds : list
        Random seeds to use.
        
    Returns
    -------
    dict
        Stability analysis results.
    """
    if seeds is None:
        seeds = list(range(10))
    
    # Get training data from explicit arg or from explainer internals
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
    
    all_weights = []
    all_rankings = []
    
    for seed in seeds:
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
    }

