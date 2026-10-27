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

Reimplementation notes:
    y_test is no longer accepted as a parameter — it was previously accepted
    but never used. Callers should remove it from their call sites.

    MIP scoring formula: avg_movement + 0.5 * max_movement.
    The 0.5 coefficient is an approximation; cross-check with the original
    MIP repository if exact replication is needed.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.base import clone
from typing import Dict, Any, List
import warnings

logger = logging.getLogger(__name__)


def _build_rank_lookup(ranking: List[str]) -> Dict[str, int]:
    """Build a feature→rank dict for O(1) position lookup (vs O(n) list.index)."""
    return {feat: idx for idx, feat in enumerate(ranking)}


def compute_shap_ranking(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                         model_type: str) -> List[str]:
    """
    Compute SHAP feature ranking for a given model.

    Delegates to the canonical implementation in shap_explainer.py to avoid
    code duplication.

    Parameters
    ----------
    model : trained classifier
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    model_type : str

    Returns
    -------
    list
        Feature names ordered by mean |SHAP| importance (descending).
    """
    from src.explainers.shap_explainer import (
        get_shap_explainer, compute_shap_values, get_global_feature_importance,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = get_shap_explainer(model, model_type, X_train, n_background=100)
        sv = compute_shap_values(explainer, X_test, model_type)

    importance = get_global_feature_importance(sv, list(X_train.columns))
    return importance['Feature'].tolist()


def calculate_nmr(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: np.ndarray,
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
    5. Calculate positional displacement of common features
    6. Normalize by the Maximum Possible Movement (MPM)
    7. Repeat until only 1 feature remains
    8. NMR = mean of normalized displacements across iterations

    Parameters
    ----------
    model : sklearn-compatible classifier
        The model (will be cloned for each iteration).
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : np.ndarray
        Training labels.
    model_type : str
    random_seed : int
        Random seed (set on model clones that support random_state).

    Returns
    -------
    dict
        NMR value, per-iteration details, and intermediate rankings.
    """
    X_train_curr = X_train.copy()
    X_test_curr = X_test.copy()

    iteration_results = []
    nmr_values = []
    all_rankings: List[List[str]] = []

    iteration = 0

    while X_train_curr.shape[1] > 1:
        NF = X_train_curr.shape[1]

        # Maximum Possible Movement (MPM) — from original NMR code
        MPM = sum(NF - j for j in range(1, NF, 2)) * 2

        if MPM == 0:
            break

        # Clone and optionally seed the model
        model_clone = clone(model)
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = random_seed

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clone.fit(X_train_curr, y_train)

        ranking = compute_shap_ranking(
            model_clone, X_train_curr, X_test_curr, model_type
        )
        all_rankings.append(ranking)

        if iteration > 0:
            prev_ranking = all_rankings[iteration - 1]
            # Use dict for O(1) position lookup instead of list.index (O(n))
            prev_lookup = _build_rank_lookup(prev_ranking)
            curr_lookup = _build_rank_lookup(ranking)

            common_features = [f for f in ranking if f in prev_lookup]

            total_movement = sum(
                abs(prev_lookup[f] - curr_lookup[f]) for f in common_features
            )

            normalized_movement = total_movement / MPM if MPM > 0 else 0.0
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

    nmr_value = float(np.mean(nmr_values)) if nmr_values else 0.0

    return {
        'nmr': nmr_value,
        'n_iterations': len(nmr_values),
        'per_iteration': iteration_results,
        'all_rankings': all_rankings,
        'nmr_values_per_iteration': nmr_values,
    }


def calculate_mip(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                  y_train: np.ndarray,
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

    MIP scoring formula:
        score(f) = mean_movement(f) + 0.5 * max_movement(f)
    where movement is positive when the feature moves up (lower rank index).
    The 0.5 coefficient is an approximation — see module docstring.

    Parameters
    ----------
    model : sklearn-compatible classifier
    X_train : pd.DataFrame
    X_test : pd.DataFrame
    y_train : np.ndarray
    model_type : str
    initial_ranking : list
        Initial feature ranking from SHAP.
    random_seed : int

    Returns
    -------
    dict
        MIP-adjusted ranking, comparison with original, and details.
    """
    X_train_curr = X_train.copy()
    X_test_curr = X_test.copy()

    feature_movements: Dict[str, List[float]] = {feat: [] for feat in initial_ranking}
    removed_features: List[str] = []
    all_rankings: List[List[str]] = [initial_ranking.copy()]

    for iteration in range(len(initial_ranking) - 1):
        if X_train_curr.shape[1] <= 1:
            break

        model_clone = clone(model)
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = random_seed

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clone.fit(X_train_curr, y_train)

        ranking = compute_shap_ranking(
            model_clone, X_train_curr, X_test_curr, model_type
        )

        if iteration > 0:
            prev_ranking = all_rankings[-1]
            # Use dict for O(1) lookup
            prev_lookup = _build_rank_lookup(prev_ranking)
            curr_lookup = _build_rank_lookup(ranking)

            for feat in ranking:
                if feat in prev_lookup:
                    movement = prev_lookup[feat] - curr_lookup[feat]  # positive = moved up
                    feature_movements[feat].append(float(movement))

        all_rankings.append(ranking)

        # Remove top feature
        top_feature = ranking[0]
        removed_features.append(top_feature)
        X_train_curr = X_train_curr.drop(columns=[top_feature])
        X_test_curr = X_test_curr.drop(columns=[top_feature])

    # Calculate MIP score
    mip_scores: Dict[str, float] = {}
    for feat in initial_ranking:
        movements = feature_movements[feat]
        if movements:
            avg_movement = float(np.mean(movements))
            max_movement = float(np.max(movements))
            mip_scores[feat] = avg_movement + 0.5 * max_movement
        else:
            # Features removed early get score based on initial position
            initial_pos = initial_ranking.index(feat)
            mip_scores[feat] = float(len(initial_ranking) - initial_pos)

    # MIP-adjusted ranking (descending MIP score)
    mip_ranking = sorted(mip_scores.keys(), key=lambda f: mip_scores[f], reverse=True)

    # Comparison table
    comparison = []
    initial_lookup = _build_rank_lookup(initial_ranking)
    mip_lookup = _build_rank_lookup(mip_ranking)
    for feat in initial_ranking:
        comparison.append({
            'Feature': feat,
            'SHAP_Rank': initial_lookup[feat] + 1,
            'MIP_Rank': mip_lookup[feat] + 1,
            'Rank_Change': (initial_lookup[feat] + 1) - (mip_lookup[feat] + 1),
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
                            y_train: np.ndarray,
                            random_seed: int = 42) -> Dict[str, Dict]:
    """
    Compute NMR for all four models.

    Paper-reported NMR values (Table in Section 3):
    - LGBM: 0.231
    - LR:   0.275
    - DT:   0.445
    - SVC:  0.273

    Parameters
    ----------
    model_results : dict
        Results from train_and_evaluate_all().
    X_train, X_test : pd.DataFrame
    y_train : np.ndarray
    random_seed : int

    Returns
    -------
    dict
        NMR results for each model.
    """
    paper_nmr = {'LGBM': 0.231, 'LR': 0.275, 'DT': 0.445, 'SVC': 0.273}

    nmr_results: Dict[str, Dict] = {}

    for model_type, res in model_results.items():
        print(f"\nComputing NMR for {res['name']} ({model_type})...")
        print(f"  (Iteratively retraining on subsets of features — may take a while)")

        nmr = calculate_nmr(
            res['model'], X_train, X_test, y_train,
            model_type, random_seed,
        )

        nmr['paper_value'] = paper_nmr.get(model_type)
        if nmr['paper_value'] is not None:
            nmr['difference'] = abs(nmr['nmr'] - nmr['paper_value'])

        nmr_results[model_type] = nmr
        print(f"  NMR = {nmr['nmr']:.3f} "
              f"(paper: {paper_nmr.get(model_type, 'N/A')})")

    return nmr_results
