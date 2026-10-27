"""
Ranking stability and comparison metrics.

Provides metrics for evaluating the stability and agreement of
feature rankings across models and runs, including:
- Spearman rank correlation
- Kendall's tau
- Jaccard similarity (top-k overlap)

Performance note:
    All functions use dict-based O(1) rank lookup instead of list.index()
    to avoid O(n²) complexity in loops.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from typing import Dict, List, Any


def _rank_lookup(ranking: List[str]) -> Dict[str, int]:
    """Build feature→position dict for O(1) rank access."""
    return {feat: idx for idx, feat in enumerate(ranking)}


def spearman_rank_correlation(ranking1: List[str], ranking2: List[str]) -> float:
    """
    Calculate Spearman rank correlation between two feature rankings.

    Parameters
    ----------
    ranking1, ranking2 : list
        Ordered lists of feature names (rank 0 = most important).

    Returns
    -------
    float
        Spearman correlation coefficient (-1 to 1).
    """
    common = [f for f in ranking1 if f in set(ranking2)]
    if len(common) < 2:
        return 0.0

    lookup1 = _rank_lookup(ranking1)
    lookup2 = _rank_lookup(ranking2)

    ranks1 = [lookup1[f] for f in common]
    ranks2 = [lookup2[f] for f in common]

    corr, _ = spearmanr(ranks1, ranks2)
    return float(corr)


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
    common = [f for f in ranking1 if f in set(ranking2)]
    if len(common) < 2:
        return 0.0

    lookup1 = _rank_lookup(ranking1)
    lookup2 = _rank_lookup(ranking2)

    ranks1 = [lookup1[f] for f in common]
    ranks2 = [lookup2[f] for f in common]

    tau, _ = kendalltau(ranks1, ranks2)
    return float(tau)


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

    Only the upper triangle (i < j) is computed to avoid duplicates.
    Self-comparisons (i == j) are excluded.

    Parameters
    ----------
    rankings : dict
        Model name -> feature ranking list.

    Returns
    -------
    pd.DataFrame
        Pairwise comparison table with Spearman, Kendall, and Jaccard metrics.
    """
    models = list(rankings.keys())
    results = []

    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i < j:
                r1, r2 = rankings[m1], rankings[m2]
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

    Computes all pairwise Spearman, Kendall, and Jaccard@5 scores across
    the provided rankings and summarises them.

    Parameters
    ----------
    all_rankings : list of lists
        Feature rankings from multiple runs.

    Returns
    -------
    dict
        Stability metrics (mean/std/min/max Spearman, Kendall, Jaccard@5).
    """
    n = len(all_rankings)
    if n < 2:
        return {
            'mean_spearman': 1.0, 'std_spearman': 0.0,
            'mean_kendall': 1.0, 'std_kendall': 0.0,
            'mean_jaccard_5': 1.0,
        }

    spearman_vals = []
    kendall_vals = []
    jaccard_vals = []

    for i in range(n):
        for j in range(i + 1, n):
            spearman_vals.append(spearman_rank_correlation(all_rankings[i], all_rankings[j]))
            kendall_vals.append(kendall_tau(all_rankings[i], all_rankings[j]))
            jaccard_vals.append(jaccard_top_k(all_rankings[i], all_rankings[j], k=5))

    return {
        'mean_spearman': float(np.mean(spearman_vals)),
        'std_spearman': float(np.std(spearman_vals)),
        'min_spearman': float(np.min(spearman_vals)),
        'max_spearman': float(np.max(spearman_vals)),
        'mean_kendall': float(np.mean(kendall_vals)),
        'std_kendall': float(np.std(kendall_vals)),
        'mean_jaccard_5': float(np.mean(jaccard_vals)),
        'std_jaccard_5': float(np.std(jaccard_vals)),
        'n_comparisons': len(spearman_vals),
    }
