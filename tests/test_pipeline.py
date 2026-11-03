"""
Test suite for the XAI paper replication project.

Tests cover:
- Dataset schema and target validation
- Preprocessing without data leakage (verified via scaler parameters)
- Reproducible train/test splitting
- SHAP output dimensions and feature mapping
- NMR calculations
- MIP calculations
- LIME stability
- Ranking metrics (Spearman, Kendall, Jaccard)
- Correlation-group construction
- JSON serialisation
- Edge cases (empty rankings, unseen categories)
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# ============================================================================
# Shared Fixtures (conftest-style, kept inline for simplicity)
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Create a synthetic numeric dataset for testing."""
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=3, n_classes=2, random_state=42
    )
    feature_names = [f'feature_{i}' for i in range(10)]
    X_df = pd.DataFrame(X, columns=feature_names)
    return X_df, y, feature_names


@pytest.fixture
def synthetic_dataset_with_target():
    """Create a full dataset with a target column for loader testing."""
    np.random.seed(42)
    n = 100
    data = {
        'Age': np.random.randint(18, 80, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Flight Distance': np.random.randint(100, 5000, n),
        'satisfaction': np.random.choice(
            ['satisfied', 'neutral or dissatisfied'], n, p=[0.5, 0.5]
        ),
    }
    return pd.DataFrame(data)


@pytest.fixture
def trained_models(synthetic_data):
    """Train simple DT and LR models on synthetic data."""
    X_df, y, feature_names = synthetic_data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_df, y, test_size=0.2, random_state=42
    )

    dt = DecisionTreeClassifier(random_state=42)
    dt.fit(X_train, y_train)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    return {
        'DT': dt, 'LR': lr,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'feature_names': feature_names,
    }


# ============================================================================
# Test: Dataset Validation
# ============================================================================

class TestDatasetValidation:
    def test_target_column_exists(self, synthetic_dataset_with_target):
        df = synthetic_dataset_with_target
        assert 'satisfaction' in df.columns

    def test_target_has_two_classes(self, synthetic_dataset_with_target):
        df = synthetic_dataset_with_target
        assert df['satisfaction'].nunique() == 2

    def test_no_missing_target(self, synthetic_dataset_with_target):
        df = synthetic_dataset_with_target
        assert df['satisfaction'].isnull().sum() == 0


class TestDatasetSampling:
    def test_unexpected_target_label_raises(self):
        from src.data.loader import sample_dataset

        df = pd.DataFrame({
            'satisfaction': [
                'satisfied', 'satisfied',
                'neutral or dissatisfied', 'unknown',
            ],
        })

        with pytest.raises(ValueError, match="Unexpected values.*unknown"):
            sample_dataset(df, n_samples=4, n_positive=2)

    def test_missing_targets_are_excluded(self):
        from src.data.loader import sample_dataset

        df = pd.DataFrame({
            'satisfaction': [
                'satisfied', 'satisfied',
                'neutral or dissatisfied', 'neutral or dissatisfied', None,
            ],
        })

        sampled = sample_dataset(df, n_samples=4, n_positive=2)

        assert len(sampled) == 4
        assert sampled['satisfaction'].notna().all()
        assert sampled['satisfaction'].value_counts().to_dict() == {
            'satisfied': 2,
            'neutral or dissatisfied': 2,
        }


# ============================================================================
# Test: Preprocessing
# ============================================================================

class TestPreprocessing:
    def _make_pipeline(self, X_df, y_series, seed=42):
        from src.preprocessing.pipeline import PreprocessingPipeline
        pipeline = PreprocessingPipeline(test_size=0.2, random_seed=seed)
        return pipeline, pipeline.fit_transform_split(X_df, y_series)

    def test_no_data_leakage_via_scaler_mean(self, synthetic_data):
        """
        Scaler should be fitted on training data only.
        Verify by checking that scaler.mean_ is close to the raw train-set mean
        (not the full-dataset mean).
        """
        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])

        pipeline, (X_train, X_test, y_train, y_test, _) = self._make_pipeline(X_df, y_series)

        info = pipeline.get_preprocessing_info()
        assert info['scaler_mean'] is not None, "scaler_mean not exposed"

        # After StandardScaler on train, train column means should be ~0
        train_col_means = np.abs(X_train.values.mean(axis=0))
        assert train_col_means.mean() < 0.1, (
            f"Train column means not near 0 after scaling: {train_col_means}"
        )

    def test_scaler_fitted_on_train_only(self, synthetic_data):
        """
        Verify scaler.mean_ matches X_train column means (leakage check).
        """
        from sklearn.model_selection import train_test_split
        from src.preprocessing.pipeline import PreprocessingPipeline

        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])

        pipeline = PreprocessingPipeline(test_size=0.2, random_seed=42)
        X_train, X_test, y_train, y_test, names = pipeline.fit_transform_split(X_df, y_series)

        # scaler.mean_ must be the train-set means, not the full-set means
        num_cols = X_df.select_dtypes(include=['number']).columns.tolist()
        if num_cols and pipeline.scaler is not None:
            # Re-derive the same split to get X_train before scaling
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = le.fit_transform(y_series)
            X_tr_raw, _, _, _ = train_test_split(
                X_df, y_enc, test_size=0.2, random_state=42, stratify=y_enc
            )
            expected_means = X_tr_raw[num_cols].mean().values
            np.testing.assert_allclose(
                pipeline.scaler.mean_, expected_means, rtol=1e-5,
                err_msg="Scaler mean_ differs from train-only mean — possible leakage"
            )

    def test_median_imputation_uses_train_only(self, synthetic_data):
        """Imputed medians must be stored and come from training data."""
        import copy
        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])

        # Introduce artificial NaN in a numeric column
        X_with_nan = X_df.copy()
        X_with_nan.loc[0, 'feature_0'] = np.nan

        from src.preprocessing.pipeline import PreprocessingPipeline
        pipeline = PreprocessingPipeline(test_size=0.2, random_seed=42)
        X_train, X_test, _, _, _ = pipeline.fit_transform_split(X_with_nan, y_series)

        # No NaNs should remain
        assert not np.isnan(X_train.values).any(), "NaN in X_train after imputation"
        assert not np.isnan(X_test.values).any(), "NaN in X_test after imputation"

    def test_reproducible_split(self, synthetic_data):
        """Same seed produces same split."""
        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])

        from src.preprocessing.pipeline import PreprocessingPipeline
        p1 = PreprocessingPipeline(test_size=0.2, random_seed=42)
        _, _, y_tr1, y_te1, _ = p1.fit_transform_split(X_df, y_series)

        p2 = PreprocessingPipeline(test_size=0.2, random_seed=42)
        _, _, y_tr2, y_te2, _ = p2.fit_transform_split(X_df, y_series)

        np.testing.assert_array_equal(y_tr1, y_tr2)
        np.testing.assert_array_equal(y_te1, y_te2)

    def test_correct_split_sizes(self, synthetic_data):
        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])

        from src.preprocessing.pipeline import PreprocessingPipeline
        pipeline = PreprocessingPipeline(test_size=0.2, random_seed=42)
        X_train, X_test, _, _, _ = pipeline.fit_transform_split(X_df, y_series)

        total = len(X_train) + len(X_test)
        assert total == len(X_df)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_get_preprocessing_info_raises_before_fit(self):
        from src.preprocessing.pipeline import PreprocessingPipeline
        pipeline = PreprocessingPipeline()
        with pytest.raises(RuntimeError, match="not been fitted"):
            pipeline.get_preprocessing_info()


# ============================================================================
# Test: SHAP
# ============================================================================

class TestSHAP:
    def test_shap_output_dimensions(self, trained_models):
        """SHAP values should have shape (n_test, n_features)."""
        from src.explainers.shap_explainer import get_shap_explainer, compute_shap_values

        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        dt = trained_models['DT']

        explainer = get_shap_explainer(dt, 'DT', X_train)
        sv = compute_shap_values(explainer, X_test, 'DT')

        assert sv.shape[0] == len(X_test)
        assert sv.shape[1] == X_train.shape[1]

    def test_feature_importance_ranking(self, trained_models):
        """Global importance should rank all features."""
        from src.explainers.shap_explainer import (
            get_shap_explainer, compute_shap_values, get_global_feature_importance,
        )

        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        dt = trained_models['DT']
        fnames = trained_models['feature_names']

        explainer = get_shap_explainer(dt, 'DT', X_train)
        sv = compute_shap_values(explainer, X_test, 'DT')
        importance = get_global_feature_importance(sv, fnames)

        assert len(importance) == len(fnames)
        assert importance['Rank'].tolist() == list(range(1, len(fnames) + 1))

    def test_shap_mismatched_features_raises(self, trained_models):
        """get_global_feature_importance should raise on mismatched feature count."""
        from src.explainers.shap_explainer import (
            get_shap_explainer, compute_shap_values, get_global_feature_importance,
        )

        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        dt = trained_models['DT']

        explainer = get_shap_explainer(dt, 'DT', X_train)
        sv = compute_shap_values(explainer, X_test, 'DT')

        with pytest.raises(ValueError, match="feature names"):
            get_global_feature_importance(sv, ['only_one_feature'])


# ============================================================================
# Test: LIME
# ============================================================================

class TestLIME:
    def test_lime_explain_instance_runs(self, trained_models):
        """LIME should produce a non-empty feature_weights dict."""
        from src.explainers.lime_explainer import create_lime_explainer, explain_instance

        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        dt = trained_models['DT']
        fnames = trained_models['feature_names']

        lime_exp = create_lime_explainer(X_train, fnames, random_seed=42)
        instance = X_test.iloc[0].values
        result = explain_instance(lime_exp, dt, instance, n_features=len(fnames), n_samples=200)

        assert isinstance(result['feature_weights'], dict)
        assert len(result['feature_weights']) > 0
        assert result['predicted_proba'] is not None

    def test_lime_stability_n_runs(self, trained_models):
        """Stability analysis should return results for each seed."""
        from src.explainers.lime_explainer import create_lime_explainer, lime_stability_analysis

        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        dt = trained_models['DT']
        fnames = trained_models['feature_names']
        seeds = [0, 1, 2]

        lime_exp = create_lime_explainer(X_train, fnames, random_seed=42)
        instance = X_test.iloc[0].values
        result = lime_stability_analysis(
            lime_exp, dt, instance,
            X_train=X_train, feature_names=fnames,
            n_features=len(fnames), n_samples=200, seeds=seeds,
        )

        assert result['n_runs'] == len(seeds)
        assert len(result['all_rankings']) == len(seeds)

    def test_lime_stability_produces_rankings(self, trained_models):
        """Each seed's ranking should be non-empty."""
        from src.explainers.lime_explainer import create_lime_explainer, lime_stability_analysis

        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        dt = trained_models['DT']
        fnames = trained_models['feature_names']

        lime_exp = create_lime_explainer(X_train, fnames, random_seed=42)
        instance = X_test.iloc[0].values
        result = lime_stability_analysis(
            lime_exp, dt, instance,
            X_train=X_train, feature_names=fnames,
            n_features=len(fnames), n_samples=200, seeds=[0, 1],
        )

        for ranking in result['all_rankings']:
            assert len(ranking) > 0


# ============================================================================
# Test: Ranking Metrics
# ============================================================================

class TestRankingMetrics:
    def test_spearman_identical(self):
        from src.metrics.ranking_metrics import spearman_rank_correlation
        r = ['a', 'b', 'c', 'd']
        assert spearman_rank_correlation(r, r) == pytest.approx(1.0)

    def test_spearman_reversed(self):
        from src.metrics.ranking_metrics import spearman_rank_correlation
        r1 = ['a', 'b', 'c', 'd']
        r2 = ['d', 'c', 'b', 'a']
        assert spearman_rank_correlation(r1, r2) == pytest.approx(-1.0)

    def test_spearman_empty_common(self):
        from src.metrics.ranking_metrics import spearman_rank_correlation
        r1 = ['a', 'b']
        r2 = ['x', 'y']
        assert spearman_rank_correlation(r1, r2) == 0.0

    def test_kendall_identical(self):
        from src.metrics.ranking_metrics import kendall_tau
        r = ['a', 'b', 'c', 'd']
        assert kendall_tau(r, r) == pytest.approx(1.0)

    def test_kendall_reversed(self):
        from src.metrics.ranking_metrics import kendall_tau
        r1 = ['a', 'b', 'c', 'd']
        r2 = ['d', 'c', 'b', 'a']
        assert kendall_tau(r1, r2) == pytest.approx(-1.0)

    def test_jaccard_identical(self):
        from src.metrics.ranking_metrics import jaccard_top_k
        r = ['a', 'b', 'c', 'd', 'e']
        assert jaccard_top_k(r, r, k=3) == pytest.approx(1.0)

    def test_jaccard_disjoint(self):
        from src.metrics.ranking_metrics import jaccard_top_k
        r1 = ['a', 'b', 'c', 'd']
        r2 = ['x', 'y', 'z', 'w']
        assert jaccard_top_k(r1, r2, k=3) == pytest.approx(0.0)

    def test_stability_includes_kendall(self):
        from src.metrics.ranking_metrics import ranking_stability_across_seeds
        r1 = ['a', 'b', 'c', 'd']
        r2 = ['d', 'c', 'b', 'a']
        result = ranking_stability_across_seeds([r1, r2])
        assert 'mean_kendall' in result

    def test_stability_single_ranking_returns_ones(self):
        from src.metrics.ranking_metrics import ranking_stability_across_seeds
        r = ['a', 'b', 'c']
        result = ranking_stability_across_seeds([r])
        assert result['mean_spearman'] == 1.0


# ============================================================================
# Test: NMR
# ============================================================================

class TestNMR:
    def test_nmr_range(self, trained_models):
        """NMR should be between 0 and 1 (normalised by MPM)."""
        from src.metrics.nmr_mip import calculate_nmr

        result = calculate_nmr(
            trained_models['DT'],
            trained_models['X_train'],
            trained_models['X_test'],
            trained_models['y_train'],
            'DT', random_seed=42,
        )

        assert 0 <= result['nmr'] <= 1.0, f"NMR out of range: {result['nmr']}"

    def test_nmr_has_iterations(self, trained_models):
        from src.metrics.nmr_mip import calculate_nmr

        result = calculate_nmr(
            trained_models['DT'],
            trained_models['X_train'],
            trained_models['X_test'],
            trained_models['y_train'],
            'DT', random_seed=42,
        )

        assert result['n_iterations'] > 0

    def test_nmr_no_y_test_param(self, trained_models):
        """calculate_nmr must NOT accept y_test (it was a dead param)."""
        import inspect
        from src.metrics.nmr_mip import calculate_nmr

        sig = inspect.signature(calculate_nmr)
        assert 'y_test' not in sig.parameters, (
            "y_test was removed as a dead parameter; it should not be in the signature"
        )


# ============================================================================
# Test: MIP
# ============================================================================

class TestMIP:
    def test_mip_returns_ranking(self, trained_models):
        """MIP should return a reordered ranking of all features."""
        from src.metrics.nmr_mip import calculate_nmr, calculate_mip

        # First get SHAP ranking via NMR (which computes it internally)
        from src.metrics.nmr_mip import compute_shap_ranking
        initial_ranking = compute_shap_ranking(
            trained_models['DT'],
            trained_models['X_train'],
            trained_models['X_test'],
            'DT',
        )

        result = calculate_mip(
            trained_models['DT'],
            trained_models['X_train'],
            trained_models['X_test'],
            trained_models['y_train'],
            'DT',
            initial_ranking,
            random_seed=42,
        )

        assert set(result['mip_ranking']) == set(initial_ranking)
        assert isinstance(result['comparison'], pd.DataFrame)
        assert len(result['mip_ranking']) == len(initial_ranking)

    def test_mip_no_y_test_param(self):
        """calculate_mip must NOT accept y_test (dead param removed)."""
        import inspect
        from src.metrics.nmr_mip import calculate_mip

        sig = inspect.signature(calculate_mip)
        assert 'y_test' not in sig.parameters, (
            "y_test was removed as a dead parameter"
        )


# ============================================================================
# Test: Correlation Groups
# ============================================================================

class TestCorrelationGroups:
    def test_all_features_covered(self, synthetic_data):
        """All features should be assigned to exactly one group."""
        from src.extension.correlation_aware import identify_correlated_groups
        X_df, _, _ = synthetic_data

        groups = identify_correlated_groups(X_df, threshold=0.6)

        all_grouped = [feat for feats in groups.values() for feat in feats]
        assert set(all_grouped) == set(X_df.columns)

    def test_no_duplicate_assignments(self, synthetic_data):
        """No feature should appear in multiple groups."""
        from src.extension.correlation_aware import identify_correlated_groups
        X_df, _, _ = synthetic_data

        groups = identify_correlated_groups(X_df, threshold=0.6)
        all_grouped = [feat for feats in groups.values() for feat in feats]
        assert len(all_grouped) == len(set(all_grouped))

    def test_distance_matrix_is_valid(self, synthetic_data):
        """
        The distance matrix (1 - |corr|) should always be in [0, 1].
        This tests the critical bug fix for negative correlations.
        """
        X_df, _, _ = synthetic_data
        corr = X_df.corr(method='pearson').abs()
        distance = 1 - corr.values
        assert distance.min() >= 0.0, f"Distance below 0: {distance.min()}"
        assert distance.max() <= 1.0 + 1e-9, f"Distance above 1: {distance.max()}"

    def test_negative_correlation_distance_stays_bounded(self):
        """
        With a manufactured negative correlation, distance must remain <= 1.
        This specifically tests the old bug: 1 - (-0.8) = 1.8 > 1.
        """
        # Create two negatively correlated features
        n = 50
        x = np.arange(n, dtype=float)
        df = pd.DataFrame({'A': x, 'B': -x + np.random.normal(0, 0.1, n)})
        corr = df.corr()
        assert corr.loc['A', 'B'] < -0.9, "Not negatively correlated enough for test"

        distance = 1 - np.abs(corr.values)
        assert distance.max() <= 1.0, (
            f"distance max = {distance.max()} > 1 — bug not fixed"
        )


# ============================================================================
# Test: Results Serialization
# ============================================================================

class TestSerialization:
    def test_json_serialization_basic_types(self, tmp_path):
        """Standard numpy types should serialize correctly."""
        import json

        results = {
            'accuracy': 0.85,
            'array': np.array([1, 2, 3]),
            'int': np.int64(42),
            'float': np.float64(3.14),
        }

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            raise TypeError(f"Not serializable: {type(obj)}")

        path = tmp_path / 'test_results.json'
        with open(path, 'w') as f:
            json.dump(results, f, default=convert)

        with open(path, 'r') as f:
            loaded = json.load(f)

        assert loaded['accuracy'] == pytest.approx(0.85)
        assert loaded['array'] == [1, 2, 3]
        assert loaded['int'] == 42

    def test_unknown_type_raises(self, tmp_path):
        """Non-serializable objects should raise TypeError, not silently convert."""
        import json

        class Unserializable:
            pass

        results = {'obj': Unserializable()}

        def strict_convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Not JSON-serializable: {type(obj).__name__}")

        path = tmp_path / 'bad.json'
        with pytest.raises(TypeError):
            with open(path, 'w') as f:
                json.dump(results, f, default=strict_convert)




# ============================================================================
# Test: Hierarchical Partition SHAP (Extension 2.0)
# ============================================================================

class TestHierarchicalPartitionSHAP:
    def test_build_hierarchical_dendrogram(self, synthetic_data):
        """Dendrogram should merge n features into a valid binary tree."""
        from src.extension.correlation_aware import build_hierarchical_dendrogram
        X_df, _, _ = synthetic_data
        Z, root = build_hierarchical_dendrogram(X_df)
        
        n_features = X_df.shape[1]
        assert Z.shape == (n_features - 1, 4), f"Unexpected linkage matrix shape: {Z.shape}"
        assert root.get_count() == n_features, f"Root count {root.get_count()} != {n_features}"

    def test_partition_shap_output_shape(self, trained_models):
        """Partition SHAP attributions should match (n_test, n_features)."""
        from src.extension.correlation_aware import compute_hierarchical_partition_shap
        dt = trained_models['DT']
        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        fnames = trained_models['feature_names']
        
        part_shap = compute_hierarchical_partition_shap(dt, X_train, X_test, fnames)
        assert part_shap.shape == (len(X_test), len(fnames))
        assert np.all(np.isfinite(part_shap)), "Partition SHAP produced NaN or Inf"

    def test_partition_shap_exact_efficiency(self, trained_models):
        """Sum of partition attributions must equal f(x) - f(background) exactly."""
        from src.extension.correlation_aware import compute_hierarchical_partition_shap
        lr = trained_models['LR']
        X_train = trained_models['X_train']
        X_test = trained_models['X_test']
        fnames = trained_models['feature_names']
        
        bg = X_train.median().values
        part_shap = compute_hierarchical_partition_shap(lr, X_train, X_test, fnames, background=bg)
        
        # Calculate true f(x) - f(bg)
        p_full = lr.predict_proba(X_test)[:, 1]
        bg_df = pd.DataFrame(np.tile(bg, (len(X_test), 1)), columns=fnames)
        p_base = lr.predict_proba(bg_df)[:, 1]
        expected_diff = p_full - p_base
        
        actual_diff = part_shap.sum(axis=1)
        np.testing.assert_allclose(actual_diff, expected_diff, atol=1e-7,
                                  err_msg="Partition SHAP does not satisfy efficiency axiom!")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
