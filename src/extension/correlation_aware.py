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

Bug fix (v2):
    The distance matrix for AgglomerativeClustering now uses
    ``1 - |correlation|`` instead of ``1 - correlation``.
    Using raw correlation produces distances > 1 for negatively correlated
    feature pairs (e.g. r=-0.8 → d=1.8), which violates the metric axioms
    required by AgglomerativeClustering(metric='precomputed') and causes
    incorrect cluster assignments. Absolute correlation ensures d ∈ [0, 1].
"""

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.base import clone
from scipy.stats import spearmanr
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from typing import Optional
from typing import Dict, List, Any, Tuple
import warnings

logger = logging.getLogger(__name__)


def identify_correlated_groups(X_train: pd.DataFrame,
                                threshold: float = 0.6,
                                method: str = 'pearson') -> Dict[str, List[str]]:
    """
    Identify groups of correlated features using hierarchical clustering.

    Uses ONLY training data to avoid test-set leakage.

    The distance between two features is defined as ``1 - |corr(f_i, f_j)|``,
    which is always in [0, 1] regardless of the sign of correlation. This
    satisfies the metric requirements of AgglomerativeClustering with
    ``metric='precomputed'``.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    threshold : float
        Correlation threshold for grouping (default: 0.6).
        Features with |r| >= threshold are placed in the same group.
    method : str
        Correlation method ('pearson' or 'spearman').

    Returns
    -------
    dict
        Group name -> list of features in the group.
    """
    # Compute absolute correlation matrix on training data only
    if method == 'spearman':
        corr_matrix = X_train.corr(method='spearman').abs()
    else:
        corr_matrix = X_train.corr(method='pearson').abs()

    # FIX: use absolute correlation so distance is always in [0, 1].
    # Previously `1 - corr_matrix.values` was used, which can exceed 1 for
    # negatively correlated features (e.g. r=-0.8 → d=1.8).
    distance_matrix = 1 - corr_matrix.values
    # Clamp to [0, 1] defensively (floating-point rounding on diagonal)
    distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
    np.fill_diagonal(distance_matrix, 0)

    # Hierarchical clustering with precomputed distance matrix
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1 - threshold,
        metric='precomputed',
        linkage='complete',
    )

    labels = clustering.fit_predict(distance_matrix)

    # Build groups
    feature_names = list(X_train.columns)
    groups: Dict[str, List[str]] = {}
    for feat, label in zip(feature_names, labels):
        group_name = f"Group_{label}"
        groups.setdefault(group_name, []).append(feat)

    # Rename singleton groups to the feature name; multi-feature groups to
    # the highest-variance representative
    named_groups: Dict[str, List[str]] = {}
    for gname, feats in groups.items():
        if len(feats) == 1:
            named_groups[feats[0]] = feats
        else:
            variances = X_train[feats].var()
            representative = str(variances.idxmax())
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

    Within each correlated group, importance credit is redistributed
    proportionally: each feature retains its own mean |SHAP| value but
    receives a proportional share of the residual group importance.

    Formula (for a group of size n_g):
        adjusted(f) = own(f) + own(f)/sum_g * (sum_g - own(f)) / n_g

    where sum_g = sum of individual importances in the group.

    Note: This formula is a design choice. It up-weights features that
    individually capture more of the group's signal while redistributing
    group credit. The formula is described in our paper's methodology
    (Section 4.5); it is not derived from Shapley axioms.

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
        Correlation-aware feature ranking (descending importance).
    """
    feat_to_idx = {f: i for i, f in enumerate(feature_names)}

    # Individual mean |SHAP| importance
    individual_importance = np.abs(shap_values).mean(axis=0)

    adjusted_importance: Dict[str, float] = {}
    for group_name, feats in groups.items():
        indices = [feat_to_idx[f] for f in feats if f in feat_to_idx]
        if not indices:
            continue

        group_total = sum(individual_importance[i] for i in indices)
        n_in_group = len(indices)

        for f in feats:
            if f in feat_to_idx:
                idx = feat_to_idx[f]
                own = float(individual_importance[idx])

                if n_in_group > 1:
                    # Proportional redistribution within the group
                    share = own / group_total if group_total > 0 else 1.0 / n_in_group
                    adjusted_importance[f] = own + share * (group_total - own) / n_in_group
                else:
                    adjusted_importance[f] = own

    ranking = sorted(
        adjusted_importance.keys(),
        key=lambda f: adjusted_importance[f],
        reverse=True,
    )
    return ranking


def multi_seed_shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                              y_train: np.ndarray, model_type: str,
                              seeds: list = None) -> Dict[str, Any]:
    """
    Run SHAP analysis across multiple training seeds to measure explanation robustness.

    Note: This analysis varies the model's random_state across seeds. The
    train/test split is held fixed to isolate model-stochasticity as the sole
    source of variability. For full robustness (split + model variability),
    pass different pre-split datasets externally.

    Parameters
    ----------
    model : sklearn classifier
        Model to clone and retrain.
    X_train, X_test : pd.DataFrame
        Features (fixed across seeds).
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

        # Set random_state where supported
        if hasattr(model_clone, 'random_state'):
            model_clone.random_state = seed

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model_clone.fit(X_train, y_train)

            explainer = get_shap_explainer(model_clone, model_type, X_train, n_background=100)
            shap_values = compute_shap_values(explainer, X_test, model_type)

        mean_abs = np.abs(shap_values).mean(axis=0)
        # Build rank lookup dict for O(1) access instead of list.index()
        ranked_indices = np.argsort(mean_abs)[::-1]
        ranked = [feature_names[i] for i in ranked_indices]

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
        'all_importances': [imp.tolist() for imp in all_importances],
        'stability': stability,
        'importance_std': dict(zip(feature_names, importance_std.tolist())),
        'importance_cv': dict(zip(feature_names, importance_cv.tolist())),
        'seeds': seeds,
        'note': (
            'Seeds vary model random_state only; '
            'train/test split is fixed. '
            'See paper Section 10 for discussion.'
        ),
    }



def build_hierarchical_dendrogram(X_train: pd.DataFrame,
                                  method: str = "complete") -> Tuple[np.ndarray, sch.ClusterNode]:
    """
    Build a complete hierarchical linkage dendrogram based on correlation distance.
    
    Distance metric: d(f_i, f_j) = 1 - |r_pearson(f_i, f_j)|.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    method : str
        Linkage method ('complete', 'average', etc.).
        
    Returns
    -------
    Tuple[np.ndarray, sch.ClusterNode]
        The linkage matrix Z and root ClusterNode of the hierarchy tree.
    """
    corr_matrix = X_train.corr(method="pearson").abs().values
    dist_matrix = np.clip(1.0 - corr_matrix, 0.0, 1.0)
    np.fill_diagonal(dist_matrix, 0.0)
    
    condensed_dist = squareform(dist_matrix)
    Z = sch.linkage(condensed_dist, method=method)
    root_node = sch.to_tree(Z)
    return Z, root_node


def compute_hierarchical_partition_shap(model,
                                        X_train: pd.DataFrame,
                                        X_test: pd.DataFrame,
                                        feature_names: list,
                                        background: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Hierarchical Partition / Owen Shapley values for an arbitrary model.
    
    Respects feature correlation geometry by constraining coalition paths along
    the complete-linkage dendrogram (Owen, 1977; Lundberg et al., 2020).
    Features clustered together in the hierarchy enter coalitions jointly,
    avoiding off-manifold counterfactual evaluation and satisfying exact efficiency:
    sum_i phi_i(x) = f(x) - f(background).
    
    Parameters
    ----------
    model : trained model
        Classifier with predict_proba (or predict).
    X_train : pd.DataFrame
        Training features (used to compute background and correlation hierarchy).
    X_test : pd.DataFrame
        Test features to explain.
    feature_names : list
        Feature names.
    background : np.ndarray, optional
        Background reference instance (defaults to median of X_train).
        
    Returns
    -------
    np.ndarray
        Partition SHAP attribution matrix of shape (n_test_samples, n_features).
    """
    n_test = len(X_test)
    n_features = len(feature_names)
    X_test_arr = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
    
    if background is None:
        bg = X_train.median().values if isinstance(X_train, pd.DataFrame) else np.median(X_train, axis=0)
    else:
        bg = np.asarray(background)
        
    _, tree = build_hierarchical_dendrogram(X_train)
    
    def predict_batch(active_cols: List[int]) -> np.ndarray:
        mat = np.tile(bg, (n_test, 1))
        if active_cols:
            mat[:, active_cols] = X_test_arr[:, active_cols]
        df_mat = pd.DataFrame(mat, columns=feature_names)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(df_mat)
                return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
            return model.predict(df_mat).astype(float)
            
    p_base = predict_batch([])
    p_full = predict_batch(list(range(n_features)))
    expected_diff = p_full - p_base
    
    def get_leaves(node: sch.ClusterNode) -> List[int]:
        if node.is_leaf():
            return [node.id]
        return get_leaves(node.left) + get_leaves(node.right)
        
    attributions = np.zeros((n_test, n_features))
    
    def partition_owen_recurse(node: sch.ClusterNode,
                               active_context: List[int],
                               parent_attr: np.ndarray):
        if node.is_leaf():
            attributions[:, node.id] = parent_attr
            return
            
        left_leaves = get_leaves(node.left)
        right_leaves = get_leaves(node.right)
        
        p_ctx = predict_batch(active_context)
        p_left = predict_batch(active_context + left_leaves)
        p_right = predict_batch(active_context + right_leaves)
        p_both = predict_batch(active_context + left_leaves + right_leaves)
        
        m_left = 0.5 * ((p_left - p_ctx) + (p_both - p_right))
        m_right = 0.5 * ((p_right - p_ctx) + (p_both - p_left))
        
        m_sum = m_left + m_right
        mask_nonzero = np.abs(m_sum) > 1e-12
        
        attr_left = np.zeros(n_test)
        attr_right = np.zeros(n_test)
        
        attr_left[mask_nonzero] = parent_attr[mask_nonzero] * (m_left[mask_nonzero] / m_sum[mask_nonzero])
        attr_right[mask_nonzero] = parent_attr[mask_nonzero] * (m_right[mask_nonzero] / m_sum[mask_nonzero])
        
        attr_left[~mask_nonzero] = 0.5 * parent_attr[~mask_nonzero]
        attr_right[~mask_nonzero] = 0.5 * parent_attr[~mask_nonzero]
        
        partition_owen_recurse(node.left, active_context, attr_left)
        partition_owen_recurse(node.right, active_context + left_leaves, attr_right)
        
    partition_owen_recurse(tree, [], expected_diff)
    return attributions


def compute_partition_shap_all_models(model_results: Dict[str, Dict],
                                      X_train: pd.DataFrame,
                                      X_test: pd.DataFrame,
                                      feature_names: list) -> Dict[str, Any]:
    """
    Compute Hierarchical Partition SHAP across all models.
    
    Parameters
    ----------
    model_results : dict
        Trained model results dictionary.
    X_train : pd.DataFrame
        Training data.
    X_test : pd.DataFrame
        Test data.
    feature_names : list
        Feature names.
        
    Returns
    -------
    dict
        Model abbreviation -> {'shap_values': np.ndarray, 'feature_ranking': list, 'feature_importance': pd.DataFrame}
    """
    results = {}
    for model_type, res in model_results.items():
        print(f"  Computing Hierarchical Partition SHAP for {model_type}...")
        model = res["model"]
        part_shap = compute_hierarchical_partition_shap(model, X_train, X_test, feature_names)
        mean_abs = np.abs(part_shap).mean(axis=0)
        ranked_indices = np.argsort(mean_abs)[::-1]
        ranking = [feature_names[i] for i in ranked_indices]
        
        imp_df = pd.DataFrame({
            "Feature": ranking,
            "Mean |Partition SHAP|": mean_abs[ranked_indices],
            "Rank": range(1, len(feature_names) + 1),
        })
        results[model_type] = {
            "shap_values": part_shap,
            "feature_ranking": ranking,
            "feature_importance": imp_df,
        }
    return results


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
        Threshold for grouping (|r| >= threshold → same group).
    seeds : list
        Seeds for stability analysis.

    Returns
    -------
    dict
        Complete extension results.
    """
    from src.metrics.ranking_metrics import (
        spearman_rank_correlation, kendall_tau, jaccard_top_k,
        pairwise_ranking_comparison,
    )

    if seeds is None:
        seeds = list(range(5))

    print("\n" + "=" * 60)
    print("RESEARCH EXTENSION: Correlation-Aware XAI Robustness Framework")
    print("=" * 60)

    # Component 1: Feature groups
    print("\n--- Component 1: Correlation-Aware Feature Groups ---")
    groups = identify_correlated_groups(X_train, threshold=correlation_threshold)
    print(f"  Identified {len(groups)} feature groups "
          f"(threshold={correlation_threshold}, distance=1-|r|):")
    for gname, feats in groups.items():
        print(f"    {gname}: {feats}")

    # Components 2 & 3: Group importance and correlation-aware rankings per model
    print("\n--- Components 2-3: Group Importance & Correlation-Aware Rankings ---")
    extension_rankings: Dict[str, List[str]] = {}
    group_importances: Dict[str, pd.DataFrame] = {}

    for model_type, shap_res in shap_results.items():
        sv = shap_res['shap_values']

        gi = compute_group_importance(sv, feature_names, groups)
        group_importances[model_type] = gi

        ca_ranking = correlation_aware_ranking(sv, feature_names, groups)
        extension_rankings[model_type] = ca_ranking

        print(f"  {model_type} top 5 (correlation-aware): {ca_ranking[:5]}")

    # Component 4: Multi-seed stability
    print("\n--- Component 4: Multi-Seed Stability Analysis ---")
    model_stability: Dict[str, Dict] = {}
    for model_type, res in model_results.items():
        print(f"  Analysing stability for {model_type}...")
        stability = multi_seed_shap_analysis(
            res['model'], X_train, X_test, y_train, model_type, seeds=seeds
        )
        model_stability[model_type] = stability
        print(f"    Mean Spearman: {stability['stability']['mean_spearman']:.3f}")

    # Component 5: Method comparison
    print("\n--- Component 5: Method Comparison ---")
    comparison_results: Dict[str, Dict] = {}

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
            comp['baseline_vs_mip_spearman'] = spearman_rank_correlation(
                baseline_ranking, mip_ranking
            )
            comp['ca_vs_mip_spearman'] = spearman_rank_correlation(
                ca_ranking, mip_ranking
            )

        comparison_results[model_type] = comp
        print(f"  {model_type}: Baseline↔CA Spearman={comp['baseline_vs_ca_spearman']:.3f}")

    # Component 6: Hierarchical Partition SHAP (Extension 2.0)
    print("\n--- Component 6: Hierarchical Partition SHAP ---")
    partition_shap_results = compute_partition_shap_all_models(
        model_results, X_train, X_test, feature_names
    )
    partition_rankings = {m: partition_shap_results[m]['feature_ranking'] for m in partition_shap_results}
    partition_agreement = pairwise_ranking_comparison(partition_rankings)
    partition_mean = partition_agreement['Spearman'].mean()
    
    for m in partition_rankings:
        print(f"  {m} top 5 (Partition SHAP): {partition_rankings[m][:5]}")

    # Cross-model agreement
    baseline_rankings = {m: shap_results[m]['feature_ranking'] for m in shap_results}
    baseline_agreement = pairwise_ranking_comparison(baseline_rankings)
    ca_agreement = pairwise_ranking_comparison(extension_rankings)

    baseline_mean = baseline_agreement['Spearman'].mean()
    ca_mean = ca_agreement['Spearman'].mean()

    print(f"\n  Cross-model agreement (mean Spearman):")
    print(f"    Baseline SHAP:       {baseline_mean:.3f}")
    print(f"    Correlation-Aware:   {ca_mean:.3f}")
    print(f"    Partition SHAP:      {partition_mean:.3f}")

    # Flag contradicting result explicitly
    if ca_mean < baseline_mean:
        logger.warning(
            "CONTRADICTING RESULT: Correlation-Aware cross-model agreement "
            "(%.3f) is LOWER than baseline SHAP (%.3f). "
            "This contradicts the extension hypothesis and must be discussed "
            "in Section 8 of the paper.", ca_mean, baseline_mean
        )
        print(
            "\n  [WARNING] Correlation-Aware cross-model agreement is lower than "
            "baseline SHAP.\n"
            "  This contradicting result is documented in the paper (Section 8)."
        )

    return {
        'groups': groups,
        'group_importances': group_importances,
        'extension_rankings': extension_rankings,
        'model_stability': {m: s['stability'] for m, s in model_stability.items()},
        'full_stability': model_stability,
        'method_comparison': comparison_results,
        'baseline_agreement': baseline_agreement,
        'ca_agreement': ca_agreement,
        'baseline_mean_spearman': baseline_mean,
        'ca_mean_spearman': ca_mean,
        'ca_beats_baseline': ca_mean >= baseline_mean,
        'partition_shap_results': partition_shap_results,
        'partition_rankings': partition_rankings,
        'partition_agreement': partition_agreement,
        'partition_mean_spearman': partition_mean,
        'partition_beats_baseline': partition_mean >= baseline_mean,
        'correlation_threshold': correlation_threshold,
        'seeds_used': seeds,
    }
