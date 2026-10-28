"""
Publication-quality visualization module.

Generates all figures for the paper replication and extension:
- Correlation heatmap (reproduces Figure 4)
- SHAP global importance plots (reproduces Figure 2)
- Cross-model ranking comparison
- LIME local explanation plots (reproduces Figure 3)
- NMR comparison
- SHAP vs MIP ranking comparison (reproduces Table 3)
- Extension stability plots
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from pathlib import Path
from typing import Dict, List, Optional
import warnings


# Publication style settings
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'figure.figsize': (10, 8),
})


def ensure_dir(path: str):
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)


def plot_correlation_heatmap(X: pd.DataFrame, save_path: str = None,
                              title: str = 'Feature Correlation Heatmap'):
    """
    Plot feature correlation heatmap (reproduces Figure 4 of the paper).
    
    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    save_path : str, optional
        Path to save the figure.
    title : str
        Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    
    corr_matrix = X.corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt='.2f',
        cmap='RdBu_r',
        center=0,
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={'size': 7}
    )
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)
    return corr_matrix


def plot_shap_global_importance(shap_results: Dict[str, Dict],
                                 model_results: Dict[str, Dict],
                                 save_path: str = None):
    """
    Plot global SHAP feature importance for all models side by side.
    Reproduces Figure 2 of the paper.
    
    Parameters
    ----------
    shap_results : dict
        SHAP results for each model.
    model_results : dict
        Model results with accuracy.
    save_path : str, optional
        Path to save.
    """
    n_models = len(shap_results)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 10))
    
    if n_models == 1:
        axes = [axes]
    
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63']
    
    for idx, (model_type, shap_res) in enumerate(shap_results.items()):
        ax = axes[idx]
        
        importance = shap_res['feature_importance']
        acc = model_results[model_type]['metrics']['accuracy']
        
        # Bar plot
        y_pos = range(len(importance))
        ax.barh(y_pos, importance['Mean |SHAP|'].values[::-1], 
                color=colors[idx % len(colors)], alpha=0.8, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance['Feature'].values[::-1], fontsize=9)
        ax.set_xlabel('Mean |SHAP value|')
        ax.set_title(f"{model_type} (ACC: {acc:.2f})", fontsize=13, fontweight='bold')
        ax.invert_yaxis()
    
    fig.suptitle('SHAP Global Feature Importance Across Models', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def plot_ranking_comparison(shap_results: Dict[str, Dict], save_path: str = None):
    """
    Visualize feature ranking changes across models with bump chart.
    
    Parameters
    ----------
    shap_results : dict
        SHAP results.
    save_path : str, optional
        Path to save.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    
    models = list(shap_results.keys())
    all_features = shap_results[models[0]]['feature_ranking']
    n_features = len(all_features)
    
    # Create rank matrix
    rank_matrix = {}
    for feat in all_features:
        ranks = []
        for model in models:
            ranking = shap_results[model]['feature_ranking']
            if feat in ranking:
                ranks.append(ranking.index(feat) + 1)
            else:
                ranks.append(n_features)
        rank_matrix[feat] = ranks
    
    # Color map
    cmap = matplotlib.colormaps['tab20'].resampled(n_features)
    
    for i, (feat, ranks) in enumerate(rank_matrix.items()):
        color = cmap(i)
        ax.plot(range(len(models)), ranks, '-o', color=color, 
                linewidth=1.5, markersize=6, label=feat, alpha=0.7)
    
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylabel('Rank (1 = most important)', fontsize=12)
    ax.set_title('Feature Ranking Comparison Across Models', 
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def plot_shap_summary(shap_values: np.ndarray, X_test: pd.DataFrame,
                       feature_names: list, model_name: str,
                       save_path: str = None):
    """
    Generate SHAP beeswarm/summary plot for a single model.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values.
    X_test : pd.DataFrame
        Test data.
    feature_names : list
        Feature names.
    model_name : str
        Model name for title.
    save_path : str, optional
        Path to save.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shap.summary_plot(
            shap_values, X_test, feature_names=feature_names,
            show=False, plot_size=None
        )
    
    plt.title(f'SHAP Summary - {model_name}', fontsize=14, fontweight='bold')
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.close('all')


def plot_shap_local(shap_values: np.ndarray, X_test: pd.DataFrame,
                    instance_idx: int, feature_names: list,
                    model_name: str, save_path: str = None):
    """
    Plot SHAP waterfall for a single instance.
    
    Parameters
    ----------
    shap_values : np.ndarray
        SHAP values for all test instances.
    X_test : pd.DataFrame
        Test data.
    instance_idx : int
        Index of the instance to explain.
    feature_names : list
        Feature names.
    model_name : str
        Model name.
    save_path : str, optional
        Path to save.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    instance_shap = shap_values[instance_idx]
    
    # Sort by absolute value
    sorted_idx = np.argsort(np.abs(instance_shap))[::-1]
    sorted_features = [feature_names[i] for i in sorted_idx]
    sorted_values = instance_shap[sorted_idx]
    
    colors = ['#E91E63' if v > 0 else '#2196F3' for v in sorted_values]
    
    y_pos = range(len(sorted_features))
    ax.barh(y_pos, sorted_values, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_features, fontsize=9)
    ax.set_xlabel('SHAP Value')
    ax.set_title(f'SHAP Local Explanation - {model_name}\n(Instance {instance_idx})',
                 fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def plot_lime_comparison(lime_results: Dict[str, Dict], 
                          instance_idx: int, save_path: str = None):
    """
    Plot LIME explanations for the same instance across all models.
    Reproduces Figure 3 of the paper.
    
    Parameters
    ----------
    lime_results : dict
        LIME results for each model.
    instance_idx : int
        Instance index.
    save_path : str, optional
        Path to save.
    """
    n_models = len(lime_results)
    fig, axes = plt.subplots(n_models, 1, figsize=(14, 4 * n_models))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_type, lime_res) in enumerate(lime_results.items()):
        ax = axes[idx]
        
        weights = lime_res['feature_weights']
        features = list(weights.keys())
        values = list(weights.values())
        
        # Sort by absolute value
        sorted_pairs = sorted(zip(features, values), key=lambda x: abs(x[1]), reverse=True)
        features = [p[0] for p in sorted_pairs]
        values = [p[1] for p in sorted_pairs]
        
        colors = ['#E91E63' if v > 0 else '#2196F3' for v in values]
        
        y_pos = range(len(features))
        ax.barh(y_pos, values, color=colors, alpha=0.8, edgecolor='white')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Feature Weight')
        
        proba = lime_res['predicted_proba']
        proba_str = f"P(Satisfied)={proba[1]:.3f}" if len(proba) > 1 else ""
        ax.set_title(f'{model_type} — LIME Local Explanation (Instance {instance_idx}) — {proba_str}',
                     fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('LIME Explanations Across Models (Same Instance)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def plot_nmr_comparison(nmr_results: Dict[str, Dict], save_path: str = None):
    """
    Plot NMR values for all models, comparing with paper-reported values.
    
    Parameters
    ----------
    nmr_results : dict
        NMR results for each model.
    save_path : str, optional
        Path to save.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = list(nmr_results.keys())
    our_nmr = [nmr_results[m]['nmr'] for m in models]
    paper_nmr = [nmr_results[m].get('paper_value', 0) for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, paper_nmr, width, label='Paper Reported',
                   color='#2196F3', alpha=0.8, edgecolor='white')
    bars2 = ax.bar(x + width/2, our_nmr, width, label='Our Reproduction',
                   color='#FF9800', alpha=0.8, edgecolor='white')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('NMR Value')
    ax.set_title('NMR Comparison: Paper vs. Reproduction', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12)
    ax.legend()
    ax.set_ylim(0, max(max(our_nmr), max(paper_nmr)) * 1.2)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def plot_shap_vs_mip(mip_results: Dict[str, Any], save_path: str = None):
    """
    Plot SHAP vs MIP ranking comparison (reproduces Table 3 visually).
    
    Parameters
    ----------
    mip_results : dict
        MIP results including comparison DataFrame.
    save_path : str, optional
        Path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    
    comparison = mip_results['comparison']
    
    # Left: Bump chart showing rank changes
    ax = axes[0]
    for _, row in comparison.iterrows():
        color = '#E91E63' if row['Rank_Change'] > 0 else '#2196F3' if row['Rank_Change'] < 0 else '#666666'
        ax.plot([0, 1], [row['SHAP_Rank'], row['MIP_Rank']], '-o', 
                color=color, linewidth=1.5, markersize=5, alpha=0.7)
        ax.annotate(row['Feature'], (0, row['SHAP_Rank']), 
                    textcoords="offset points", xytext=(-10, 0),
                    ha='right', fontsize=8)
        ax.annotate(row['Feature'], (1, row['MIP_Rank']),
                    textcoords="offset points", xytext=(10, 0),
                    ha='left', fontsize=8)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['SHAP Rank', 'MIP Rank'], fontsize=12)
    ax.set_ylabel('Rank')
    ax.set_title('SHAP vs MIP Feature Rankings', fontsize=13, fontweight='bold')
    ax.invert_yaxis()
    
    # Right: Rank change bar chart
    ax2 = axes[1]
    features = comparison['Feature'].values
    changes = comparison['Rank_Change'].values
    colors = ['#4CAF50' if c > 0 else '#E91E63' if c < 0 else '#666666' for c in changes]
    
    y_pos = range(len(features))
    ax2.barh(y_pos, changes, color=colors, alpha=0.8, edgecolor='white')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(features, fontsize=9)
    ax2.set_xlabel('Rank Change (positive = moved up in MIP)')
    ax2.set_title('Rank Changes: SHAP → MIP', fontsize=13, fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.suptitle('SHAP vs MIP Ranking Comparison (Collinearity-Adjusted)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def plot_classifier_performance(model_results: Dict[str, Dict], save_path: str = None):
    """
    Plot classifier performance comparison.
    
    Parameters
    ----------
    model_results : dict
        Results from train_and_evaluate_all().
    save_path : str, optional
        Path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    models = list(model_results.keys())
    metrics_names = ['accuracy', 'precision', 'recall', 'f1_score']
    
    # Left: Performance metrics
    ax = axes[0]
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics_names):
        values = [model_results[m]['metrics'][metric] for m in models]
        ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(),
               alpha=0.8, edgecolor='white')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('Score')
    ax.set_title('Classifier Performance Comparison', fontsize=13, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(models)
    ax.legend()
    ax.set_ylim(0, 1.1)
    
    # Right: Our accuracy vs paper accuracy
    ax2 = axes[1]
    our_acc = [model_results[m]['metrics']['accuracy'] for m in models]
    paper_acc = [model_results[m]['metrics'].get('paper_accuracy', 0) for m in models]
    
    bars1 = ax2.bar(x - 0.2, paper_acc, 0.4, label='Paper', color='#2196F3', alpha=0.8)
    bars2 = ax2.bar(x + 0.2, our_acc, 0.4, label='Ours', color='#FF9800', alpha=0.8)
    
    ax2.set_xlabel('Model')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy: Paper vs. Reproduction', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()
    ax2.set_ylim(0.5, 1.0)
    
    for bar in bars1:
        ax2.annotate(f'{bar.get_height():.2f}', 
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax2.annotate(f'{bar.get_height():.2f}',
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points",
                     ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)


def plot_extension_stability(stability_results: Dict, save_path: str = None):
    """
    Plot stability comparison for the research extension.
    
    Parameters
    ----------
    stability_results : dict
        Results from extension experiments.
    save_path : str, optional
        Path to save.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    if 'model_stability' in stability_results:
        # Plot per-model stability
        ax = axes[0]
        models = list(stability_results['model_stability'].keys())
        spearman_vals = [stability_results['model_stability'][m].get('mean_spearman', 0) 
                         for m in models]
        
        ax.bar(models, spearman_vals, color=['#2196F3', '#4CAF50', '#FF9800', '#E91E63'],
               alpha=0.8, edgecolor='white')
        ax.set_ylabel('Mean Spearman Correlation')
        ax.set_title('Explanation Stability per Model', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1.1)
        
        for i, v in enumerate(spearman_vals):
            ax.annotate(f'{v:.3f}', (i, v), textcoords="offset points",
                        xytext=(0, 5), ha='center', fontsize=10)
    
    if 'method_comparison' in stability_results:
        ax2 = axes[1]
        comp = stability_results['method_comparison']
        methods = list(comp.keys())
        values = [
            v.get('baseline_vs_ca_spearman', 0.0) if isinstance(v, dict) else float(v)
            for v in comp.values()
        ]
        
        ax2.bar(methods, values, color='#9C27B0', alpha=0.8, edgecolor='white')
        ax2.set_ylabel('Baseline vs CA Spearman')
        ax2.set_title('Method Comparison (Spearman)', fontsize=13, fontweight='bold')
        ax2.set_ylim(0, 1.1)
        for i, v in enumerate(values):
            ax2.annotate(f'{v:.3f}', (i, v), textcoords="offset points",
                         xytext=(0, 5), ha='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir(str(Path(save_path).parent))
        fig.savefig(save_path)
        print(f"Saved: {save_path}")
    
    plt.close(fig)
