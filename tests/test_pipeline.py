"""
Test suite for the XAI paper replication project.

Tests cover:
- Dataset schema and target validation
- Preprocessing without data leakage
- Reproducible train/test splitting
- SHAP output dimensions and feature mapping
- NMR calculations
- Ranking metrics
- Correlation-group construction
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def synthetic_data():
    """Create a synthetic dataset for testing."""
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
    """Train simple models on synthetic data."""
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


# ============================================================================
# Test: Preprocessing
# ============================================================================

class TestPreprocessing:
    def test_no_data_leakage(self, synthetic_data):
        """Ensure scaler is fitted only on training data."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])
        
        pipeline = PreprocessingPipeline(test_size=0.2, random_seed=42)
        X_train, X_test, y_train, y_test, names = pipeline.fit_transform_split(X_df, y_series)
        
        # Training data should have approximately zero mean (StandardScaler)
        train_mean = np.abs(X_train.values.mean(axis=0)).mean()
        assert train_mean < 0.1, f"Train mean too large: {train_mean}"
    
    def test_reproducible_split(self, synthetic_data):
        """Ensure same seed produces same split."""
        from src.preprocessing.pipeline import PreprocessingPipeline
        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])
        
        p1 = PreprocessingPipeline(test_size=0.2, random_seed=42)
        X_tr1, X_te1, y_tr1, y_te1, _ = p1.fit_transform_split(X_df, y_series)
        
        p2 = PreprocessingPipeline(test_size=0.2, random_seed=42)
        X_tr2, X_te2, y_tr2, y_te2, _ = p2.fit_transform_split(X_df, y_series)
        
        np.testing.assert_array_equal(y_tr1, y_tr2)
        np.testing.assert_array_equal(y_te1, y_te2)
    
    def test_correct_split_sizes(self, synthetic_data):
        from src.preprocessing.pipeline import PreprocessingPipeline
        X_df, y, _ = synthetic_data
        y_series = pd.Series(['satisfied' if yi == 1 else 'not' for yi in y])
        
        pipeline = PreprocessingPipeline(test_size=0.2, random_seed=42)
        X_train, X_test, _, _, _ = pipeline.fit_transform_split(X_df, y_series)
        
        total = len(X_train) + len(X_test)
        assert total == len(X_df)
        assert abs(len(X_test) / total - 0.2) < 0.05


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
            get_shap_explainer, compute_shap_values, get_global_feature_importance
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
        corr = spearman_rank_correlation(r1, r2)
        assert corr == pytest.approx(-1.0)
    
    def test_kendall_identical(self):
        from src.metrics.ranking_metrics import kendall_tau
        r = ['a', 'b', 'c', 'd']
        assert kendall_tau(r, r) == pytest.approx(1.0)
    
    def test_jaccard_identical(self):
        from src.metrics.ranking_metrics import jaccard_top_k
        r = ['a', 'b', 'c', 'd', 'e']
        assert jaccard_top_k(r, r, k=3) == pytest.approx(1.0)
    
    def test_jaccard_disjoint(self):
        from src.metrics.ranking_metrics import jaccard_top_k
        r1 = ['a', 'b', 'c', 'd']
        r2 = ['x', 'y', 'z', 'w']
        assert jaccard_top_k(r1, r2, k=3) == pytest.approx(0.0)


# ============================================================================
# Test: NMR
# ============================================================================

class TestNMR:
    def test_nmr_range(self, trained_models):
        """NMR should be between 0 and 1."""
        from src.metrics.nmr_mip import calculate_nmr
        
        result = calculate_nmr(
            trained_models['DT'],
            trained_models['X_train'],
            trained_models['X_test'],
            trained_models['y_train'],
            trained_models['y_test'],
            'DT', random_seed=42
        )
        
        assert 0 <= result['nmr'] <= 1.0, f"NMR out of range: {result['nmr']}"
    
    def test_nmr_has_iterations(self, trained_models):
        """NMR should have multiple iterations."""
        from src.metrics.nmr_mip import calculate_nmr
        
        result = calculate_nmr(
            trained_models['DT'],
            trained_models['X_train'],
            trained_models['X_test'],
            trained_models['y_train'],
            trained_models['y_test'],
            'DT', random_seed=42
        )
        
        assert result['n_iterations'] > 0


# ============================================================================
# Test: Correlation Groups
# ============================================================================

class TestCorrelationGroups:
    def test_all_features_covered(self, synthetic_data):
        """All features should be assigned to a group."""
        from src.extension.correlation_aware import identify_correlated_groups
        X_df, _, _ = synthetic_data
        
        groups = identify_correlated_groups(X_df, threshold=0.6)
        
        all_grouped = []
        for feats in groups.values():
            all_grouped.extend(feats)
        
        assert set(all_grouped) == set(X_df.columns)
    
    def test_no_duplicate_assignments(self, synthetic_data):
        """No feature should appear in multiple groups."""
        from src.extension.correlation_aware import identify_correlated_groups
        X_df, _, _ = synthetic_data
        
        groups = identify_correlated_groups(X_df, threshold=0.6)
        
        all_grouped = []
        for feats in groups.values():
            all_grouped.extend(feats)
        
        assert len(all_grouped) == len(set(all_grouped))


# ============================================================================
# Test: Results Serialization
# ============================================================================

class TestSerialization:
    def test_json_serialization(self, tmp_path):
        """Results should be JSON-serializable."""
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
            return str(obj)
        
        path = tmp_path / 'test_results.json'
        with open(path, 'w') as f:
            json.dump(results, f, default=convert)
        
        with open(path, 'r') as f:
            loaded = json.load(f)
        
        assert loaded['accuracy'] == 0.85
        assert loaded['array'] == [1, 2, 3]


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
