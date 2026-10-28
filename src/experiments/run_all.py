"""
Main experiment runner for the XAI paper replication and extension.

Orchestrates the complete experimental pipeline:
1. Load and validate dataset
2. Preprocess
3. Train classifiers
4. SHAP analysis
5. LIME analysis
6. NMR computation
7. MIP computation
8. Research extension
9. Generate all visualizations and results tables

Usage:
    python -m src.experiments.run_all [--config configs/experiment_config.yaml]
"""

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loader import (
    load_dataset, validate_dataset, sample_dataset, get_feature_columns
)
from src.preprocessing.pipeline import PreprocessingPipeline
from src.models.classifiers import train_and_evaluate_all, create_performance_table
from src.explainers.shap_explainer import compute_all_shap, compare_rankings
from src.explainers.lime_explainer import (
    create_lime_explainer, compute_lime_for_instance, lime_stability_analysis
)
from src.metrics.nmr_mip import compute_nmr_all_models, calculate_mip
from src.metrics.ranking_metrics import pairwise_ranking_comparison
from src.visualization.plots import (
    plot_correlation_heatmap, plot_shap_global_importance,
    plot_ranking_comparison, plot_shap_summary, plot_shap_local,
    plot_lime_comparison, plot_nmr_comparison, plot_shap_vs_mip,
    plot_classifier_performance, plot_extension_stability
)
from src.extension.correlation_aware import run_extension_experiments


warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


def load_config(config_path: str = None) -> dict:
    """Load experiment configuration."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'configs' / 'experiment_config.yaml'
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results: dict, path: str):
    """Save results as JSON (converting non-serializable objects)."""
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        if isinstance(obj, datetime):
            return obj.isoformat()
        return str(obj)
    
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, default=convert, indent=2)
    print(f"Saved: {path}")


def run_full_pipeline(config: dict = None, data_path: str = None):
    """
    Run the complete experimental pipeline.
    
    Parameters
    ----------
    config : dict, optional
        Experiment configuration. If None, loads from default path.
    data_path : str, optional
        Path to dataset CSV.
    """
    start_time = time.time()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if config is None:
        config = load_config()
    
    seed = config['random_seeds']['master_seed']
    np.random.seed(seed)
    
    results_dir = PROJECT_ROOT / config['output']['results_dir']
    figures_dir = PROJECT_ROOT / config['output']['figures_dir']
    tables_dir = PROJECT_ROOT / config['output']['tables_dir']
    
    for d in [results_dir, figures_dir, tables_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'timestamp': timestamp,
        'config': config,
        'seed': seed,
    }
    
    # =========================================================================
    # STEP 1: Load and Validate Dataset
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 1: LOADING AND VALIDATING DATASET")
    print("="*70)
    
    df = load_dataset(data_path)
    validation = validate_dataset(df)
    print(f"Dataset shape: {df.shape}")
    print(f"Target column: {validation['target_column']}")
    print(f"Target distribution: {validation.get('target_distribution', 'N/A')}")
    print(f"Duplicates: {validation['duplicates']}")
    
    all_results['dataset_validation'] = {
        k: v for k, v in validation.items() 
        if not isinstance(v, (np.ndarray, pd.DataFrame))
    }
    
    # Sample to match paper's 500-subject case study
    print("\nSampling to match paper (500 subjects, 250 satisfied)...")
    df_sampled = sample_dataset(
        df, 
        n_samples=config['dataset']['n_samples'],
        n_positive=config['dataset']['n_satisfied'],
        random_seed=config['random_seeds']['sampling_seed']
    )
    
    # =========================================================================
    # STEP 2: Preprocessing
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 2: PREPROCESSING")
    print("="*70)
    
    feature_columns = get_feature_columns(df_sampled)
    target_col = validation['target_column']
    
    pipeline = PreprocessingPipeline(
        test_size=config['dataset']['test_size'],
        random_seed=config['random_seeds']['split_seed']
    )
    
    X, y = pipeline.prepare_features(df_sampled, feature_columns, target_col)
    X_train, X_test, y_train, y_test, feature_names = pipeline.fit_transform_split(X, y)
    
    prep_info = pipeline.get_preprocessing_info()
    all_results['preprocessing'] = prep_info
    print(f"Feature names: {feature_names}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    
    # =========================================================================
    # STEP 3: Correlation Heatmap (Figure 4)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 3: FEATURE CORRELATION ANALYSIS")
    print("="*70)
    
    corr_matrix = plot_correlation_heatmap(
        X_train, 
        save_path=str(figures_dir / 'correlation_heatmap.png'),
        title='Feature Correlation Heatmap (Training Data)'
    )
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = abs(corr_matrix.iloc[i, j])
            if corr_val > 0.5:
                high_corr_pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Correlation': corr_val,
                })
    
    if high_corr_pairs:
        print(f"\nHighly correlated pairs (|r| > 0.5):")
        for p in sorted(high_corr_pairs, key=lambda x: x['Correlation'], reverse=True):
            print(f"  {p['Feature 1']} ↔ {p['Feature 2']}: {p['Correlation']:.3f}")
    
    all_results['high_correlations'] = high_corr_pairs
    
    # =========================================================================
    # STEP 4: Train and Evaluate Classifiers
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 4: TRAINING AND EVALUATING CLASSIFIERS")
    print("="*70)
    
    model_results = train_and_evaluate_all(
        X_train, X_test, y_train, y_test, random_seed=seed
    )
    
    perf_table = create_performance_table(model_results)
    perf_table.to_csv(str(tables_dir / 'classifier_performance.csv'), index=False)
    print(f"\n{perf_table.to_string(index=False)}")
    
    plot_classifier_performance(model_results, str(figures_dir / 'classifier_performance.png'))
    
    all_results['classifier_performance'] = {
        abbrev: {k: v for k, v in res['metrics'].items() 
                 if k != 'confusion_matrix'}
        for abbrev, res in model_results.items()
    }
    
    # =========================================================================
    # STEP 5: SHAP Analysis (Figure 2, Table 2)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 5: SHAP ANALYSIS")
    print("="*70)
    
    shap_results = compute_all_shap(
        model_results, X_train, X_test, feature_names,
        n_background=config['shap']['n_background_samples']
    )
    
    # Global importance plots
    plot_shap_global_importance(
        shap_results, model_results,
        str(figures_dir / 'shap_global_importance.png')
    )
    
    # Ranking comparison
    ranking_table = compare_rankings(shap_results)
    ranking_table.to_csv(str(tables_dir / 'shap_ranking_comparison.csv'))
    print(f"\nSHAP Feature Ranking Comparison:")
    print(ranking_table.to_string())
    
    # Ranking bump chart
    plot_ranking_comparison(shap_results, str(figures_dir / 'ranking_comparison.png'))
    
    # Per-model SHAP summary plots
    for model_type, shap_res in shap_results.items():
        plot_shap_summary(
            shap_res['shap_values'], X_test, feature_names,
            f"{model_results[model_type]['name']} ({model_type})",
            str(figures_dir / f'shap_summary_{model_type}.png')
        )
    
    # Local SHAP explanations (instance 0)
    for model_type, shap_res in shap_results.items():
        plot_shap_local(
            shap_res['shap_values'], X_test, 0, feature_names,
            f"{model_results[model_type]['name']} ({model_type})",
            str(figures_dir / f'shap_local_{model_type}_instance0.png')
        )
    
    # Pairwise ranking agreement
    rankings = {m: shap_results[m]['feature_ranking'] for m in shap_results}
    pairwise = pairwise_ranking_comparison(rankings)
    pairwise.to_csv(str(tables_dir / 'pairwise_ranking_agreement.csv'), index=False)
    print(f"\nPairwise Ranking Agreement:")
    print(pairwise.to_string(index=False))
    
    all_results['shap_rankings'] = {
        m: shap_results[m]['feature_ranking'] for m in shap_results
    }
    
    # =========================================================================
    # STEP 6: LIME Analysis (Figure 3)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 6: LIME ANALYSIS")
    print("="*70)
    
    target_classes = pipeline.target_encoder.classes_.tolist()
    lime_exp = create_lime_explainer(
        X_train, feature_names, class_names=target_classes,
        random_seed=config['random_seeds']['lime_seed']
    )
    
    # Explain instance 0 across all models
    instance_idx = 0
    instance = X_test.iloc[instance_idx].values
    
    lime_results = compute_lime_for_instance(
        model_results, lime_exp, instance, instance_idx,
        n_features=len(feature_names),
        n_samples=config['lime']['n_samples']
    )
    
    plot_lime_comparison(lime_results, instance_idx, 
                         str(figures_dir / 'lime_comparison_instance0.png'))
    
    # LIME stability analysis for one model
    print("\nLIME Stability Analysis (LGBM, instance 0)...")
    lime_stability = lime_stability_analysis(
        lime_exp, model_results['LGBM']['model'], instance,
        X_train=X_train,
        feature_names=feature_names,
        class_names=target_classes,
        n_features=len(feature_names),
        n_samples=config['lime']['n_samples'],
        seeds=config['random_seeds']['robustness_seeds']
    )
    print(f"  Ran {lime_stability['n_runs']} LIME explanations with different seeds")
    
    all_results['lime_stability'] = {
        'n_runs': lime_stability['n_runs'],
        'seeds': lime_stability['seeds'],
    }
    
    # =========================================================================
    # STEP 7: NMR Analysis
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 7: NORMALIZED MOVEMENT RATE (NMR)")
    print("="*70)
    
    nmr_results = compute_nmr_all_models(
        model_results, X_train, X_test, y_train, y_test, random_seed=seed
    )
    
    plot_nmr_comparison(nmr_results, str(figures_dir / 'nmr_comparison.png'))
    
    nmr_table = pd.DataFrame([{
        'Model': m,
        'Our NMR': nmr_results[m]['nmr'],
        'Paper NMR': nmr_results[m].get('paper_value', None),
        'Difference': nmr_results[m].get('difference', None),
    } for m in nmr_results])
    nmr_table.to_csv(str(tables_dir / 'nmr_comparison.csv'), index=False)
    print(f"\n{nmr_table.to_string(index=False)}")
    
    all_results['nmr'] = {m: {'nmr': nmr_results[m]['nmr'], 
                               'paper': nmr_results[m].get('paper_value')}
                          for m in nmr_results}
    
    # =========================================================================
    # STEP 8: MIP Analysis (Table 3)
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 8: MODIFIED INDEX POSITION (MIP)")
    print("="*70)
    
    # Apply MIP to the model with lowest NMR (paper uses LGBM)
    best_model = min(nmr_results, key=lambda m: nmr_results[m]['nmr'])
    print(f"Best model by NMR: {best_model} (NMR={nmr_results[best_model]['nmr']:.3f})")
    
    mip_results_all = {}
    mip_result = calculate_mip(
        model_results[best_model]['model'],
        X_train, X_test, y_train, y_test,
        best_model,
        shap_results[best_model]['feature_ranking'],
        random_seed=seed
    )
    mip_results_all[best_model] = mip_result
    
    print(f"\nSHAP vs MIP Ranking (Top 10):")
    print(mip_result['comparison'].head(10).to_string(index=False))
    
    mip_result['comparison'].to_csv(str(tables_dir / 'shap_vs_mip.csv'), index=False)
    plot_shap_vs_mip(mip_result, str(figures_dir / 'shap_vs_mip.png'))
    
    all_results['mip'] = {
        'best_model': best_model,
        'initial_ranking': mip_result['initial_ranking'],
        'mip_ranking': mip_result['mip_ranking'],
    }
    
    # =========================================================================
    # STEP 9: Research Extension
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 9: RESEARCH EXTENSION")
    print("="*70)
    
    extension_results = run_extension_experiments(
        model_results, shap_results,
        X_train, X_test, y_train, y_test, feature_names,
        mip_results=mip_results_all,
        correlation_threshold=0.6,
        seeds=config['random_seeds']['robustness_seeds'][:5]
    )
    
    # Save extension results
    ext_summary = {
        'groups': {k: v for k, v in extension_results['groups'].items()},
        'model_stability': extension_results['model_stability'],
        'method_comparison': extension_results['method_comparison'],
        'correlation_threshold': extension_results['correlation_threshold'],
    }
    all_results['extension'] = ext_summary
    
    plot_extension_stability(
        extension_results, str(figures_dir / 'extension_stability.png')
    )
    
    # =========================================================================
    # STEP 10: Save All Results
    # =========================================================================
    print("\n" + "="*70)
    print("STEP 10: SAVING RESULTS")
    print("="*70)
    
    elapsed = time.time() - start_time
    all_results['runtime_seconds'] = elapsed
    
    save_results(all_results, str(results_dir / 'experiment_results.json'))
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"EXPERIMENT COMPLETE — {elapsed:.1f}s")
    print(f"{'='*70}")
    print(f"Results saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to:  {tables_dir}")
    print(f"\nKey findings:")
    print(f"  Classifier accuracies:")
    for m, r in model_results.items():
        acc = r['metrics']['accuracy']
        paper = r['metrics'].get('paper_accuracy', 'N/A')
        print(f"    {m}: {acc:.4f} (paper: {paper})")
    print(f"  NMR values:")
    for m, r in nmr_results.items():
        print(f"    {m}: {r['nmr']:.3f} (paper: {r.get('paper_value', 'N/A')})")
    print(f"  Best model by NMR: {best_model}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Run XAI paper replication experiments'
    )
    parser.add_argument('--config', type=str, default=None,
                        help='Path to experiment config YAML')
    parser.add_argument('--data', type=str, default=None,
                        help='Path to dataset CSV file')
    args = parser.parse_args()
    
    config = load_config(args.config) if args.config else load_config()
    run_full_pipeline(config, args.data)


if __name__ == '__main__':
    main()
