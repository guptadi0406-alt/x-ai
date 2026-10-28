"""
Correlation-Aware XAI Robustness Framework — Research Extension.

This module implements our proposed extension to address limitations
identified in Salih et al. (2024):
1. Feature collinearity affecting XAI rankings
2. Model dependency of explanations
3. Instability of feature importance rankings

Components:
- Correlation-aware feature grouping
- Grouped feature importance
- Cross-seed explanation robustness
- Comparison framework (Baseline SHAP/LIME vs MIP vs our extension)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import clone
from scipy.stats import spearmanr
from typing import Dict, List, Any, Tuple
import warnings


def identify_correlated_groups(X_train: pd.DataFrame, 
                                threshold: float = 0.6,
                                method: str = 'pearson') -> Dict[str, List[str]]:
    """
    Identify groups of correlated features using hierarchical clustering.
    
    Uses ONLY training data to avoid test-set leakage.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    threshold : float
        Correlation threshold for grouping (default: 0.6).
    method : str
        Correlation method ('pearson' or 'spearman').
        
    Returns
    -------
    dict
        Group name -> list of features in the group.
    """
    # Compute correlation matrix on training data only
    if method == 'spearman':
        corr_matrix = X_train.corr(method='spearman').abs()
    else:
        corr_matrix = X_train.corr(method='pearson').abs()
    
    # Convert correlation to distance
    distance_matrix = 1 - corr_matrix.values
    np.fill_diagonal(distance_matrix, 0)
    
    # Hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='precomputed',
        linkage='complete'
    )
    
    labels = clustering.fit_predict(distance_matrix)
    
    # Build groups
    feature_names = list(X_train.columns)
    groups = {}
    for feat, label in zip(feature_names, labels):
        group_name = f"Group_{label}"
        if group_name not in groups:
            groups[group_name] = []
        groups[group_name].append(feat)
    
    # Rename groups with representative feature (highest variance)
    named_groups = {}
    for gname, feats in groups.items():
        if len(feats) == 1:
            named_groups[feats[0]] = feats
        else:
            # Use the feature with highest variance as representative
            variances = X_train[feats].var()
            representative = variances.idxmax()
            named_groups[f"{representative}_group"] = feats
    
    return named_groups


def compute_group_importance(shap_values: np.ndarray, feature_names: list,
                              groups: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Aggregate SHAP importance at the group level.
    
    For correlated feature groups, sum their absolute SHAP values
    to get the total group contribution.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values matrix (n_samples, n_features).
    feature_names : list
        Feature names.
    groups : dict
        Feature groups from identify_correlated_groups().
        
    Returns
    -------
    pd.DataFrame
        Group-level importance rankings.
    """
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    
    group_importance = {}
    for group_name, feats in groups.items():
        indices = [feat_to_idx[f] for f in feats if f in feat_to_idx]
        if indices:
            # Sum absolute SHAP values for all features in the group
            group_shap = np.abs(shap_values[:, indices]).sum(axis=1).mean()
            group_importance[group_name] = {
                'importance': group_shap,
                'features': feats,
                'n_features': len(feats),
            }
    
    # Create ranked DataFrame
    rows = []
    for gname, info in sorted(group_importance.items(), 
                                key=lambda x: x[1]['importance'], reverse=True):
        rows.append({
            'Group': gname,
            'Importance': info['importance'],
            'Features': ', '.join(info['features']),
            'N_Features': info['n_features'],
        })
    
    df = pd.DataFrame(rows)
    df['Rank'] = range(1, len(df) + 1)
    return df


def correlation_aware_ranking(shap_values: np.ndarray, feature_names: list,
                               groups: Dict[str, List[str]]) -> List[str]:
    """
    Produce a correlation-aware feature ranking.
    
    Within each correlated group, features share importance credit.
    The ranking reflects both individual and group-level contributions.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values.
    feature_names : list
        Feature names.
    groups : dict
        Feature groups.
        
    Returns
    -------
    list
        Correlation-aware feature ranking.
    """
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}
    
    # Calculate individual importance
    individual_importance = np.abs(shap_values).mean(axis=0)
    
    # Adjust for group membership
    adjusted_importance = {}
    for group_name, feats in groups.items():
        indices = [feat_to_idx[f] for f in feats if f in feat_to_idx]
        if not indices:
            continue
        
        group_total = sum(individual_importance[i] for i in indices)
        n_in_group = len(indices)
        
        for f in feats:
            if f in feat_to_idx:
                idx = feat_to_idx[f]
                own = individual_importance[idx]
                
                if n_in_group > 1:
                    # Share credit: feature gets its own importance plus
                    # a fraction of group importance proportional to its share
                    share = own / group_total if group_total > 0 else 1.0 / n_in_group
                    adjusted_importance[f] = own + share * (group_total - own) / n_in_group
                else:
                    adjusted_importance[f] = own
    
    # Rank by adjusted importance
    ranking = sorted(adjusted_importance.keys(), 
                     key=lambda f: adjusted_importance[f], reverse=True)
    return ranking


def multi_seed_shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: np.ndarray, model_type: str,
                              seeds: list = None) -> Dict[str, Any]:
    """
    Run SHAP analysis across multiple training seeds to measure explanation robustness.
    
    Parameters
    ----------
    model : sklearn classifier
        Model to clone and retrain.
    X_train, X_test : pd.DataFrame
        Features.
    y_train : np.ndarray
        Labels.
    model_type : str
        Model abbreviation.
    seeds : list
        Random seeds for retraining.
        
    Returns
    -------
    dict
        Rankings from each seed and stability metrics.
    """
    from src.explainers.shap_explainer import get_shap_explainer, compute_shap_values
    from src.metrics.ranking_metrics import ranking_stability_across_seeds
    
    if seeds is None:
        seeds = list(range(10))
    
    all_rankings = []
    all_importances = []
    feature_names = list(X_train.columns)
    
    for seed in seeds:
        model_clone = clone(model)
        
        # For models that support random_state
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = seed
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clone.fit(X_train, y_train)
            
            explainer = get_shap_explainer(model_clone, model_type, X_train, n_background=100)
            shap_values = compute_shap_values(explainer, X_test, model_type)
        
        mean_abs = np.abs(shap_values).mean(axis=0)
        ranked = [feature_names[i] for i in np.argsort(mean_abs)[::-1]]
        
        all_rankings.append(ranked)
        all_importances.append(mean_abs)
    
    # Compute stability
    stability = ranking_stability_across_seeds(all_rankings)
    
    # Importance variability
    importance_matrix = np.array(all_importances)
    importance_std = importance_matrix.std(axis=0)
    importance_cv = importance_std / (importance_matrix.mean(axis=0) + 1e-10)
    
    return {
        'all_rankings': all_rankings,
        'all_importances': all_importances,
        'stability': stability,
        'importance_std': dict(zip(feature_names, importance_std)),
        'importance_cv': dict(zip(feature_names, importance_cv)),
        'seeds': seeds,
    }


def run_extension_experiments(model_results: Dict[str, Dict],
                               shap_results: Dict[str, Dict],
                               X_train: pd.DataFrame, X_test: pd.DataFrame,
                               y_train: np.ndarray, y_test: np.ndarray,
                               feature_names: list,
                               mip_results: Dict = None,
                               correlation_threshold: float = 0.6,
                               seeds: list = None) -> Dict[str, Any]:
    """
    Run the complete extension experiment suite.
    
    Components:
    1. Correlation-aware feature grouping
    2. Group-level SHAP importance
    3. Correlation-aware rankings
    4. Multi-seed stability analysis
    5. Comparison: Baseline SHAP vs MIP vs Correlation-Aware
    
    Parameters
    ----------
    model_results : dict
        Trained model results.
    shap_results : dict
        Baseline SHAP results.
    X_train, X_test : pd.DataFrame
        Features.
    y_train, y_test : np.ndarray
        Labels.
    feature_names : list
        Feature names.
    mip_results : dict, optional
        MIP results for comparison.
    correlation_threshold : float
        Threshold for grouping.
    seeds : list
        Seeds for stability analysis.
        
    Returns
    -------
    dict
        Complete extension results.
    """
    from src.metrics.ranking_metrics import (
        spearman_rank_correlation, kendall_tau, jaccard_top_k,
        pairwise_ranking_comparison
    )
    
    if seeds is None:
        seeds = list(range(5))
    
    print("\n" + "="*60)
    print("RESEARCH EXTENSION: Correlation-Aware XAI Robustness Framework")
    print("="*60)
    
    # Component 1: Feature groups
    print("\n--- Component 1: Correlation-Aware Feature Groups ---")
    groups = identify_correlated_groups(X_train, threshold=correlation_threshold)
    print(f"  Identified {len(groups)} feature groups (threshold={correlation_threshold}):")
    for gname, feats in groups.items():
        print(f"    {gname}: {feats}")
    
    # Components 2 & 3: Group importance and correlation-aware rankings per model
    print("\n--- Components 2-3: Group Importance & Correlation-Aware Rankings ---")
    extension_rankings = {}
    group_importances = {}
    
    for model_type, shap_res in shap_results.items():
        sv = shap_res['shap_values']
        
        # Group importance
        gi = compute_group_importance(sv, feature_names, groups)
        group_importances[model_type] = gi
        
        # Correlation-aware ranking
        ca_ranking = correlation_aware_ranking(sv, feature_names, groups)
        extension_rankings[model_type] = ca_ranking
        
        print(f"  {model_type} top 5 (correlation-aware): {ca_ranking[:5]}")
    
    # Component 4: Multi-seed stability
    print("\n--- Component 4: Multi-Seed Stability Analysis ---")
    model_stability = {}
    for model_type, res in model_results.items():
        print(f"  Analyzing stability for {model_type}...")
        stability = multi_seed_shap_analysis(
            res['model'], X_train, X_test, y_train, model_type, seeds=seeds
        )
        model_stability[model_type] = stability
        print(f"    Mean Spearman: {stability['stability']['mean_spearman']:.3f}")
    
    # Component 5: Method comparison
    print("\n--- Component 5: Method Comparison ---")
    comparison_results = {}
    
    for model_type in shap_results.keys():
        baseline_ranking = shap_results[model_type]['feature_ranking']
        ca_ranking = extension_rankings[model_type]
        
        comp = {
            'baseline_vs_ca_spearman': spearman_rank_correlation(baseline_ranking, ca_ranking),
            'baseline_vs_ca_kendall': kendall_tau(baseline_ranking, ca_ranking),
            'baseline_vs_ca_jaccard5': jaccard_top_k(baseline_ranking, ca_ranking, k=5),
        }
        
        if mip_results and model_type in mip_results:
            mip_ranking = mip_results[model_type]['mip_ranking']
            comp['baseline_vs_mip_spearman'] = spearman_rank_correlation(baseline_ranking, mip_ranking)
            comp['ca_vs_mip_spearman'] = spearman_rank_correlation(ca_ranking, mip_ranking)
        
        comparison_results[model_type] = comp
        print(f"  {model_type}: Baseline↔CA Spearman={comp['baseline_vs_ca_spearman']:.3f}")
    
    # Cross-model agreement for each method
    baseline_rankings = {m: shap_results[m]['feature_ranking'] for m in shap_results}
    ca_rankings_dict = extension_rankings
    
    baseline_agreement = pairwise_ranking_comparison(baseline_rankings)
    ca_agreement = pairwise_ranking_comparison(ca_rankings_dict)
    
    print(f"\n  Cross-model agreement (mean Spearman):")
    print(f"    Baseline SHAP: {baseline_agreement['Spearman'].mean():.3f}")
    print(f"    Correlation-Aware: {ca_agreement['Spearman'].mean():.3f}")
    
    return {
        'groups': groups,
        'group_importances': group_importances,
        'extension_rankings': extension_rankings,
        'model_stability': {m: s['stability'] for m, s in model_stability.items()},
        'full_stability': model_stability,
        'method_comparison': comparison_results,
        'baseline_agreement': baseline_agreement,
        'ca_agreement': ca_agreement,
        'correlation_threshold': correlation_threshold,
        'seeds_used': seeds,
    }
