"""
Ranking stability and comparison metrics.

Provides metrics for evaluating the stability and agreement of
feature rankings across models and runs, including:
- Spearman rank correlation
- Kendall's tau
- Jaccard similarity (top-k overlap)
- Standard deviation of importance values
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Tuple, Any


def spearman_rank_correlation(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate Spearman rank correlation between two feature rankings.
    
    Parameters
    ----------
    ranking1, ranking2 : list
        Ordered lists of feature names.
        
    Returns
    -------
    float
        Spearman correlation coefficient (-1 to 1).
    """
    # Find common features
    common = [f for f in ranking1 if f in ranking2]
    if len(common) < 2:
        return 0.0
    
    ranks1 = [ranking1.index(f) for f in common]
    ranks2 = [ranking2.index(f) for f in common]
    
    corr, _ = spearmanr(ranks1, ranks2)
    return corr


def kendall_tau(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate Kendall's tau correlation between two feature rankings.
    
    Parameters
    ----------
    ranking1, ranking2 : list
        Ordered lists of feature names.
        
    Returns
    -------
    float
        Kendall's tau coefficient (-1 to 1).
    """
    common = [f for f in ranking1 if f in ranking2]
    if len(common) < 2:
        return 0.0
    
    ranks1 = [ranking1.index(f) for f in common]
    ranks2 = [ranking2.index(f) for f in common]
    
    tau, _ = kendalltau(ranks1, ranks2)
    return tau


def jaccard_top_k(ranking1: List[str], ranking2: List[str], k: int = 5) -> float:
    """
    Calculate Jaccard similarity of the top-k features.
    
    Parameters
    ----------
    ranking1, ranking2 : list
        Ordered lists of feature names.
    k : int
        Number of top features to compare.
        
    Returns
    -------
    float
        Jaccard similarity (0 to 1).
    """
    top1 = set(ranking1[:k])
    top2 = set(ranking2[:k])
    
    if not top1 or not top2:
        return 0.0
    
    intersection = len(top1 & top2)
    union = len(top1 | top2)
    
    return intersection / union if union > 0 else 0.0


def pairwise_ranking_comparison(rankings: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Compute pairwise ranking comparison metrics for all model pairs.
    
    Parameters
    ----------
    rankings : dict
        Model name -> feature ranking list.
        
    Returns
    -------
    pd.DataFrame
        Pairwise comparison matrix with Spearman, Kendall, and Jaccard metrics.
    """
    models = list(rankings.keys())
    results = []
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                r1 = rankings[m1]
                r2 = rankings[m2]
                
                results.append({
                    'Model 1': m1,
                    'Model 2': m2,
                    'Spearman': spearman_rank_correlation(r1, r2),
                    'Kendall Tau': kendall_tau(r1, r2),
                    'Jaccard@3': jaccard_top_k(r1, r2, k=3),
                    'Jaccard@5': jaccard_top_k(r1, r2, k=5),
                    'Jaccard@10': jaccard_top_k(r1, r2, k=10),
                })
    
    return pd.DataFrame(results)


def ranking_stability_across_seeds(all_rankings: List[List[str]]) -> Dict[str, float]:
    """
    Measure ranking stability across multiple random seeds.
    
    Parameters
    ----------
    all_rankings : list of lists
        Feature rankings from multiple runs.
        
    Returns
    -------
    dict
        Stability metrics.
    """
    n = len(all_rankings)
    if n < 2:
        return {'mean_spearman': 1.0, 'std_spearman': 0.0, 'mean_jaccard_5': 1.0}
    
    spearman_vals = []
    jaccard_vals = []
    
    for i in range(n):
        for j in range(i + 1, n):
            spearman_vals.append(spearman_rank_correlation(all_rankings[i], all_rankings[j]))
            jaccard_vals.append(jaccard_top_k(all_rankings[i], all_rankings[j], k=5))
    
    return {
        'mean_spearman': np.mean(spearman_vals),
        'std_spearman': np.std(spearman_vals),
        'min_spearman': np.min(spearman_vals),
        'max_spearman': np.max(spearman_vals),
        'mean_jaccard_5': np.mean(jaccard_vals),
        'std_jaccard_5': np.std(jaccard_vals),
        'n_comparisons': len(spearman_vals),
    }
