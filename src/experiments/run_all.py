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
import logging
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
    create_lime_explainer, compute_lime_for_instance,
    lime_stability_analysis, lime_stability_all_models,
)
from src.metrics.nmr_mip import compute_nmr_all_models, calculate_mip
from src.metrics.ranking_metrics import pairwise_ranking_comparison
from src.visualization.plots import (
    plot_correlation_heatmap, plot_shap_global_importance,
    plot_ranking_comparison, plot_shap_summary, plot_shap_local,
    plot_lime_comparison, plot_nmr_comparison, plot_shap_vs_mip,
    plot_classifier_performance, plot_extension_stability,
)
from src.extension.correlation_aware import run_extension_experiments

# Filter only known noisy warnings; do NOT suppress all warnings globally
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = None) -> dict:
    """Load experiment configuration from YAML."""
    if config_path is None:
        config_path = PROJECT_ROOT / 'configs' / 'experiment_config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger.info("Loaded config from %s", config_path)
    return config


def save_results(results: dict, path: str):
    """
    Save results as JSON (converting non-serializable objects).

    Raises TypeError for unknown types rather than silently converting to str,
    so accidental serialization of model/explainer objects is caught early.
    """
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
        if isinstance(obj, Path):
            return str(obj)
        # For any other type, raise so the caller knows to exclude it
        raise TypeError(
            f"Object of type {type(obj).__name__} is not JSON-serializable. "
            f"Exclude it from the results dict before saving."
        )

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results, f, default=convert, indent=2)
    logger.info("Saved results to %s", path)
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

    # Read correlation threshold from config (unified — previously inconsistent)
    corr_threshold = config.get('extension', {}).get('correlation_threshold', 0.6)

    all_results = {
        'timestamp': timestamp,
        'config': config,
        'seed': seed,
    }

    # =========================================================================
    # STEP 1: Load and Validate Dataset
    # =========================================================================
    logger.info("STEP 1: LOADING AND VALIDATING DATASET")
    print("\n" + "=" * 70)
    print("STEP 1: LOADING AND VALIDATING DATASET")
    print("=" * 70)

    try:
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
    except Exception as e:
        logger.error("STEP 1 FAILED: %s", e)
        raise

    # Sample to match paper's 500-subject case study
    print("\nSampling to match paper (500 subjects, 250 satisfied)...")
    try:
        df_sampled = sample_dataset(
            df,
            n_samples=config['dataset']['n_samples'],
            n_positive=config['dataset']['n_satisfied'],
            random_seed=config['random_seeds']['sampling_seed'],
        )
    except Exception as e:
        logger.error("Sampling FAILED: %s", e)
        raise

    # =========================================================================
    # STEP 2: Preprocessing
    # =========================================================================
    logger.info("STEP 2: PREPROCESSING")
    print("\n" + "=" * 70)
    print("STEP 2: PREPROCESSING")
    print("=" * 70)

    try:
        feature_columns = get_feature_columns(df_sampled)
        target_col = validation['target_column']

        pipeline = PreprocessingPipeline(
            test_size=config['dataset']['test_size'],
            random_seed=config['random_seeds']['split_seed'],
        )

        X, y = pipeline.prepare_features(df_sampled, feature_columns, target_col)
        X_train, X_test, y_train, y_test, feature_names = pipeline.fit_transform_split(X, y)

        prep_info = pipeline.get_preprocessing_info()
        all_results['preprocessing'] = prep_info
        print(f"Feature names: {feature_names}")
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    except Exception as e:
        logger.error("STEP 2 FAILED: %s", e)
        raise

    # =========================================================================
    # STEP 3: Correlation Heatmap (Figure 4)
    # =========================================================================
    logger.info("STEP 3: FEATURE CORRELATION ANALYSIS")
    print("\n" + "=" * 70)
    print("STEP 3: FEATURE CORRELATION ANALYSIS")
    print("=" * 70)

    try:
        corr_matrix = plot_correlation_heatmap(
            X_train,
            save_path=str(figures_dir / 'correlation_heatmap.png'),
            title='Feature Correlation Heatmap (Training Data)',
        )

        # Use the unified corr_threshold from config
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > corr_threshold:
                    high_corr_pairs.append({
                        'Feature 1': corr_matrix.columns[i],
                        'Feature 2': corr_matrix.columns[j],
                        'Correlation': corr_val,
                    })

        if high_corr_pairs:
            print(f"\nHighly correlated pairs (|r| > {corr_threshold}):")
            for p in sorted(high_corr_pairs, key=lambda x: x['Correlation'], reverse=True):
                print(f"  {p['Feature 1']} ↔ {p['Feature 2']}: {p['Correlation']:.3f}")

        all_results['high_correlations'] = high_corr_pairs
        all_results['correlation_threshold_used'] = corr_threshold
    except Exception as e:
        logger.error("STEP 3 FAILED: %s", e)
        raise

    # =========================================================================
    # STEP 4: Train and Evaluate Classifiers
    # =========================================================================
    logger.info("STEP 4: TRAINING AND EVALUATING CLASSIFIERS")
    print("\n" + "=" * 70)
    print("STEP 4: TRAINING AND EVALUATING CLASSIFIERS")
    print("=" * 70)

    try:
        model_results = train_and_evaluate_all(
            X_train, X_test, y_train, y_test,
            random_seed=seed, run_cv=True,
        )

        perf_table = create_performance_table(model_results)
        perf_table.to_csv(str(tables_dir / 'classifier_performance.csv'), index=False)
        print(f"\n{perf_table.to_string(index=False)}")

        plot_classifier_performance(model_results, str(figures_dir / 'classifier_performance.png'))

        all_results['classifier_performance'] = {
            abbrev: {
                k: v for k, v in res['metrics'].items()
                if k not in ('confusion_matrix', 'classification_report')
            }
            for abbrev, res in model_results.items()
        }
    except Exception as e:
        logger.error("STEP 4 FAILED: %s", e)
        raise

    # =========================================================================
    # STEP 5: SHAP Analysis (Figure 2, Table 2)
    # =========================================================================
    logger.info("STEP 5: SHAP ANALYSIS")
    print("\n" + "=" * 70)
    print("STEP 5: SHAP ANALYSIS")
    print("=" * 70)

    try:
        shap_results = compute_all_shap(
            model_results, X_train, X_test, feature_names,
            n_background=config['shap']['n_background_samples'],
        )

        plot_shap_global_importance(
            shap_results, model_results,
            str(figures_dir / 'shap_global_importance.png'),
        )

        ranking_table = compare_rankings(shap_results)
        ranking_table.to_csv(str(tables_dir / 'shap_ranking_comparison.csv'))
        print(f"\nSHAP Feature Ranking Comparison:")
        print(ranking_table.to_string())

        plot_ranking_comparison(shap_results, str(figures_dir / 'ranking_comparison.png'))

        for model_type, shap_res in shap_results.items():
            plot_shap_summary(
                shap_res['shap_values'], X_test, feature_names,
                f"{model_results[model_type]['name']} ({model_type})",
                str(figures_dir / f'shap_summary_{model_type}.png'),
            )
            plot_shap_local(
                shap_res['shap_values'], X_test, 0, feature_names,
                f"{model_results[model_type]['name']} ({model_type})",
                str(figures_dir / f'shap_local_{model_type}_instance0.png'),
            )

        rankings = {m: shap_results[m]['feature_ranking'] for m in shap_results}
        pairwise = pairwise_ranking_comparison(rankings)
        pairwise.to_csv(str(tables_dir / 'pairwise_ranking_agreement.csv'), index=False)
        print(f"\nPairwise Ranking Agreement:")
        print(pairwise.to_string(index=False))

        all_results['shap_rankings'] = {
            m: shap_results[m]['feature_ranking'] for m in shap_results
        }
    except Exception as e:
        logger.error("STEP 5 FAILED: %s", e)
        raise

    # =========================================================================
    # STEP 6: LIME Analysis (Figure 3)
    # =========================================================================
    logger.info("STEP 6: LIME ANALYSIS")
    print("\n" + "=" * 70)
    print("STEP 6: LIME ANALYSIS")
    print("=" * 70)

    try:
        target_classes = pipeline.target_encoder.classes_.tolist()
        lime_exp = create_lime_explainer(
            X_train, feature_names, class_names=target_classes,
            random_seed=config['random_seeds']['lime_seed'],
        )

        instance_idx = 0
        instance = X_test.iloc[instance_idx].values

        lime_results = compute_lime_for_instance(
            model_results, lime_exp, instance, instance_idx,
            n_features=len(feature_names),
            n_samples=config['lime']['n_samples'],
        )

        plot_lime_comparison(lime_results, instance_idx,
                             str(figures_dir / 'lime_comparison_instance0.png'))

        # LIME stability across ALL 4 models (not just LGBM)
        print("\nLIME Stability Analysis (all models, instance 0)...")
        lime_stability_multi = lime_stability_all_models(
            model_results, X_train, feature_names, target_classes,
            instance,
            n_features=len(feature_names),
            n_samples=config['lime']['n_samples'],
            seeds=config['random_seeds']['robustness_seeds'],
        )

        all_results['lime_stability'] = {
            model_type: {
                'n_runs': res['n_runs'],
                'seeds': res['seeds'],
            }
            for model_type, res in lime_stability_multi.items()
        }
    except Exception as e:
        logger.error("STEP 6 FAILED: %s", e)
        logger.warning("Continuing pipeline despite STEP 6 failure.")
        all_results['lime_stability'] = {'error': str(e)}

    # =========================================================================
    # STEP 7: NMR Analysis
    # =========================================================================
    logger.info("STEP 7: NORMALIZED MOVEMENT RATE (NMR)")
    print("\n" + "=" * 70)
    print("STEP 7: NORMALIZED MOVEMENT RATE (NMR)")
    print("=" * 70)

    try:
        nmr_results = compute_nmr_all_models(
            model_results, X_train, X_test, y_train, random_seed=seed
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

        all_results['nmr'] = {
            m: {'nmr': nmr_results[m]['nmr'], 'paper': nmr_results[m].get('paper_value')}
            for m in nmr_results
        }
    except Exception as e:
        logger.error("STEP 7 FAILED: %s", e)
        raise

    # =========================================================================
    # STEP 8: MIP Analysis (Table 3)
    # =========================================================================
    logger.info("STEP 8: MODIFIED INDEX POSITION (MIP)")
    print("\n" + "=" * 70)
    print("STEP 8: MODIFIED INDEX POSITION (MIP)")
    print("=" * 70)

    try:
        best_model = min(nmr_results, key=lambda m: nmr_results[m]['nmr'])
        print(f"Best model by NMR: {best_model} (NMR={nmr_results[best_model]['nmr']:.3f})")

        mip_results_all = {}
        mip_result = calculate_mip(
            model_results[best_model]['model'],
            X_train, X_test, y_train,
            best_model,
            shap_results[best_model]['feature_ranking'],
            random_seed=seed,
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
    except Exception as e:
        logger.error("STEP 8 FAILED: %s", e)
        raise

    # =========================================================================
    # STEP 9: Research Extension
    # =========================================================================
    logger.info("STEP 9: RESEARCH EXTENSION")
    print("\n" + "=" * 70)
    print("STEP 9: RESEARCH EXTENSION")
    print("=" * 70)

    try:
        extension_results = run_extension_experiments(
            model_results, shap_results,
            X_train, X_test, y_train, y_test, feature_names,
            mip_results=mip_results_all,
            correlation_threshold=corr_threshold,  # unified from config
            seeds=config['random_seeds']['robustness_seeds'][:5],
        )

        ext_summary = {
            'groups': {k: v for k, v in extension_results['groups'].items()},
            'model_stability': extension_results['model_stability'],
            'method_comparison': extension_results['method_comparison'],
            'correlation_threshold': extension_results['correlation_threshold'],
            'baseline_mean_spearman': extension_results['baseline_mean_spearman'],
            'ca_mean_spearman': extension_results['ca_mean_spearman'],
            'ca_beats_baseline': extension_results['ca_beats_baseline'],
        }
        all_results['extension'] = ext_summary

        plot_extension_stability(
            extension_results, str(figures_dir / 'extension_stability.png')
        )
    except Exception as e:
        logger.error("STEP 9 FAILED: %s", e)
        logger.warning("Continuing to save partial results.")
        all_results['extension'] = {'error': str(e)}

    # =========================================================================
    # STEP 10: Save All Results
    # =========================================================================
    logger.info("STEP 10: SAVING RESULTS")
    print("\n" + "=" * 70)
    print("STEP 10: SAVING RESULTS")
    print("=" * 70)

    elapsed = time.time() - start_time
    all_results['runtime_seconds'] = elapsed

    save_results(all_results, str(results_dir / 'experiment_results.json'))

    print(f"\n{'=' * 70}")
    print(f"EXPERIMENT COMPLETE — {elapsed:.1f}s")
    print(f"{'=' * 70}")
    print(f"Results saved to: {results_dir}")
    print(f"Figures saved to: {figures_dir}")
    print(f"Tables saved to:  {tables_dir}")
    print(f"\nKey findings:")
    print(f"  Classifier accuracies:")
    for m, r in model_results.items():
        acc = r['metrics']['accuracy']
        paper = r['metrics'].get('paper_accuracy', 'N/A')
        cv_acc = r['metrics'].get('cv', {}).get('accuracy_mean', None)
        cv_str = f" | CV: {cv_acc:.4f}" if cv_acc is not None else ""
        print(f"    {m}: {acc:.4f} (paper: {paper}){cv_str}")
    print(f"  NMR values:")
    for m, r in nmr_results.items():
        print(f"    {m}: {r['nmr']:.3f} (paper: {r.get('paper_value', 'N/A')})")
    print(f"  Best model by NMR: {best_model}")

    if 'extension' in all_results and 'error' not in all_results['extension']:
        ext = all_results['extension']
        print(f"  Extension — CA vs Baseline Spearman:")
        print(f"    Baseline: {ext.get('baseline_mean_spearman', 'N/A'):.3f}")
        print(f"    CA:       {ext.get('ca_mean_spearman', 'N/A'):.3f}")
        if not ext.get('ca_beats_baseline', True):
            print("  [!] CA cross-model agreement is LOWER than baseline — see paper Section 8")

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
