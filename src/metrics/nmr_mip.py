"""
Normalized Movement Rate (NMR) and Modified Index Position (MIP) implementations.

NMR: Measures the stability of XAI feature rankings when top features are
iteratively removed. Lower NMR = more stable ranking.
Reference: Salih et al., IEEE 2022 (doi: 10.1109/ICHI54592.2022.00040)

MIP: Post-hoc method that adjusts feature rankings from any XAI method
to account for multicollinearity among features.
Reference: Salih et al., IEEE JBHI 2024

Both algorithms are reimplemented based on the original repositories:
- https://github.com/amaa11/NMR
- https://github.com/amaa11/MIP
"""

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from typing import Dict, Any, List, Tuple
import warnings


def compute_shap_ranking(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         model_type: str) -> List[str]:
    """
    Compute SHAP feature ranking for a given model.
    
    Parameters
    ----------
    model : trained classifier
        The model.
    X_train : pd.DataFrame
        Training data.
    X_test : pd.DataFrame
        Test data.
    model_type : str
        Model type abbreviation.
        
    Returns
    -------
    list
        Feature names ordered by importance (descending).
    """
    from src.explainers.shap_explainer import get_shap_explainer, compute_shap_values
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = get_shap_explainer(model, model_type, X_train, n_background=100)
        sv = compute_shap_values(explainer, X_test, model_type)
    
    mean_abs = np.abs(sv).mean(axis=0)
    # Ensure mean_abs is 1D
    mean_abs = np.asarray(mean_abs).flatten()
    feature_names = list(X_train.columns)
    ranked_indices = np.argsort(mean_abs)[::-1].tolist()
    return [feature_names[i] for i in ranked_indices]


def calculate_nmr(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: np.ndarray, y_test: np.ndarray,
                  model_type: str, random_seed: int = 42) -> Dict[str, Any]:
    """
    Calculate the Normalized Movement Rate (NMR).
    
    NMR quantifies how much the feature ranking changes when the most
    important feature is iteratively removed from the model and the model
    is retrained. A lower NMR indicates a more stable ranking.
    
    Algorithm (from the original NMR repository):
    1. Train model, compute SHAP ranking
    2. Remove the top feature
    3. Retrain model on remaining features
    4. Compute new SHAP ranking
    5. Calculate positional displacement of features
    6. Normalize by the Maximum Possible Movement (MPM)
    7. Repeat until only 1 feature remains
    8. NMR = mean of normalized displacements across iterations
    
    Parameters
    ----------
    model : sklearn-compatible classifier
        The model (will be cloned for each iteration).
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    model_type : str
        Model abbreviation.
    random_seed : int
        Random seed.
        
    Returns
    -------
    dict
        NMR value, per-iteration details, and intermediate rankings.
    """
    X_train_curr = X_train.copy()
    X_test_curr = X_test.copy()
    
    iteration_results = []
    nmr_values = []
    all_rankings = []
    
    iteration = 0
    
    while X_train_curr.shape[1] > 1:
        NF = X_train_curr.shape[1]
        
        # Calculate Maximum Possible Movement (MPM) for this iteration
        # MPM formula from original NMR code
        MPM = 0
        for j in range(1, NF, 2):
            MPM += (NF - j)
        MPM = MPM * 2
        
        if MPM == 0:
            break
        
        # Train model
        model_clone = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clone.fit(X_train_curr, y_train)
        
        # Get SHAP ranking
        ranking = compute_shap_ranking(
            model_clone, X_train_curr, X_test_curr, model_type
        )
        all_rankings.append(ranking)
        
        if iteration > 0:
            # Calculate movement between previous and current ranking
            prev_ranking = all_rankings[iteration - 1]
            # Only compare features that exist in both rankings
            common_features = [f for f in ranking if f in prev_ranking]
            
            total_movement = 0
            for feat in common_features:
                prev_pos = prev_ranking.index(feat)
                curr_pos = ranking.index(feat)
                total_movement += abs(prev_pos - curr_pos)
            
            # Normalize by MPM
            normalized_movement = total_movement / MPM if MPM > 0 else 0
            nmr_values.append(normalized_movement)
            
            iteration_results.append({
                'iteration': iteration,
                'n_features': NF,
                'removed_feature': all_rankings[iteration - 1][0],
                'total_movement': total_movement,
                'MPM': MPM,
                'normalized_movement': normalized_movement,
            })
        
        # Remove the top feature for next iteration
        top_feature = ranking[0]
        X_train_curr = X_train_curr.drop(columns=[top_feature])
        X_test_curr = X_test_curr.drop(columns=[top_feature])
        
        iteration += 1
    
    nmr_value = np.mean(nmr_values) if nmr_values else 0.0
    
    return {
        'nmr': nmr_value,
        'n_iterations': len(nmr_values),
        'per_iteration': iteration_results,
        'all_rankings': all_rankings,
        'nmr_values_per_iteration': nmr_values,
    }


def calculate_mip(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: np.ndarray, y_test: np.ndarray,
                  model_type: str, initial_ranking: List[str],
                  random_seed: int = 42) -> Dict[str, Any]:
    """
    Calculate the Modified Index Position (MIP).
    
    MIP adjusts the feature ranking from any XAI method to account for
    multicollinearity. It works by:
    1. Starting with the initial XAI ranking
    2. Removing the top feature
    3. Retraining and recomputing the ranking
    4. Observing how remaining features shift positions
    5. Using the positional shifts to create a collinearity-aware ranking
    
    Algorithm (from the original MIP repository):
    - For each iteration, remove top feature, retrain, get new ranking
    - Track how features move when correlated features are removed
    - Features that jump up significantly when a correlated feature is removed
      are re-ranked higher in the MIP-adjusted list
    
    Parameters
    ----------
    model : sklearn-compatible classifier
        The model.
    X_train : pd.DataFrame
        Training features.
    X_test : pd.DataFrame
        Test features.
    y_train : np.ndarray
        Training labels.
    y_test : np.ndarray
        Test labels.
    model_type : str
        Model abbreviation.
    initial_ranking : list
        Initial feature ranking from SHAP.
    random_seed : int
        Random seed.
        
    Returns
    -------
    dict
        MIP-adjusted ranking, comparison with original, and details.
    """
    X_train_curr = X_train.copy()
    X_test_curr = X_test.copy()
    
    # Track feature position changes
    feature_movements = {feat: [] for feat in initial_ranking}
    removed_features = []
    all_rankings = [initial_ranking.copy()]
    
    current_features = list(X_train_curr.columns)
    
    for iteration in range(len(initial_ranking) - 1):
        if X_train_curr.shape[1] <= 1:
            break
        
        # Get current SHAP ranking
        model_clone = clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clone.fit(X_train_curr, y_train)
        
        ranking = compute_shap_ranking(
            model_clone, X_train_curr, X_test_curr, model_type
        )
        
        if iteration > 0:
            prev_ranking = all_rankings[-1]
            for feat in ranking:
                if feat in prev_ranking:
                    prev_pos = prev_ranking.index(feat)
                    curr_pos = ranking.index(feat)
                    movement = prev_pos - curr_pos  # positive = moved up
                    feature_movements[feat].append(movement)
        
        all_rankings.append(ranking)
        
        # Remove top feature
        top_feature = ranking[0]
        removed_features.append(top_feature)
        X_train_curr = X_train_curr.drop(columns=[top_feature])
        X_test_curr = X_test_curr.drop(columns=[top_feature])
    
    # Calculate MIP score for each feature
    # Features that consistently move up when correlated features are removed
    # get a higher MIP score
    mip_scores = {}
    for feat in initial_ranking:
        movements = feature_movements[feat]
        if movements:
            # Average upward movement indicates the feature was suppressed
            # by collinear features
            avg_movement = np.mean(movements)
            max_movement = np.max(movements) if movements else 0
            mip_scores[feat] = avg_movement + 0.5 * max_movement
        else:
            # Top features removed early get score based on initial position
            initial_pos = initial_ranking.index(feat)
            mip_scores[feat] = len(initial_ranking) - initial_pos
    
    # Create MIP-adjusted ranking (sort by MIP score, descending)
    mip_ranking = sorted(mip_scores.keys(), key=lambda f: mip_scores[f], reverse=True)
    
    # Create comparison table
    comparison = []
    for i, feat in enumerate(initial_ranking):
        mip_pos = mip_ranking.index(feat) + 1
        comparison.append({
            'Feature': feat,
            'SHAP_Rank': i + 1,
            'MIP_Rank': mip_pos,
            'Rank_Change': (i + 1) - mip_pos,
            'MIP_Score': mip_scores[feat],
        })
    
    return {
        'initial_ranking': initial_ranking,
        'mip_ranking': mip_ranking,
        'mip_scores': mip_scores,
        'comparison': pd.DataFrame(comparison),
        'removed_features_order': removed_features,
        'feature_movements': feature_movements,
    }


def compute_nmr_all_models(model_results: Dict[str, Dict],
                            X_train: pd.DataFrame, X_test: pd.DataFrame,
                            y_train: np.ndarray, y_test: np.ndarray,
                            random_seed: int = 42) -> Dict[str, Dict]:
    """
    Compute NMR for all four models.
    
    Paper-reported NMR values (Table in Section 3):
    - LGBM: 0.231
    - LR: 0.275
    - DT: 0.445
    - SVC: 0.273
    
    Parameters
    ----------
    model_results : dict
        Results from train_and_evaluate_all().
    X_train, X_test : pd.DataFrame
        Features.
    y_train, y_test : np.ndarray
        Labels.
    random_seed : int
        Random seed.
        
    Returns
    -------
    dict
        NMR results for each model.
    """
    # Paper-reported values for comparison
    paper_nmr = {'LGBM': 0.231, 'LR': 0.275, 'DT': 0.445, 'SVC': 0.273}
    
    nmr_results = {}
    
    for model_type, res in model_results.items():
        print(f"\nComputing NMR for {res['name']} ({model_type})...")
        
        nmr = calculate_nmr(
            res['model'], X_train, X_test, y_train, y_test,
            model_type, random_seed
        )
        
        nmr['paper_value'] = paper_nmr.get(model_type)
        if nmr['paper_value'] is not None:
            nmr['difference'] = abs(nmr['nmr'] - nmr['paper_value'])
        
        nmr_results[model_type] = nmr
        
        print(f"  NMR = {nmr['nmr']:.3f} (paper: {paper_nmr.get(model_type, 'N/A')})")
    
    return nmr_results
