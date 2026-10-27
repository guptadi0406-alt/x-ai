"""
Classifier training and evaluation module.

Implements the four classifiers from the paper:
- Decision Tree (DT)
- Logistic Regression (LR)
- LightGBM (LGBM)
- Support Vector Classifier (SVC)

Documented deviations from strict "default parameters":
    LR:  max_iter=1000 (sklearn default is 100; 100 is insufficient for
         convergence on this dataset and causes ConvergenceWarning).
    SVC: probability=True (sklearn default is False; required so that
         LIME's predict_proba interface and ROC-AUC computation work).
         This activates Platt scaling internally and may marginally affect
         decision-boundary accuracy.

Both deviations are intentional and necessary for the experimental pipeline.
The paper states "default parameters were used", which we interpret as
"default hyperparameters" (C, kernel, criterion, etc.) rather than
implementation flags needed for XAI compatibility.

Cross-validation note:
    With n=500 (80/20 split → ~100 test samples) accuracy estimates carry
    a 95% CI of approximately ±4%. We report single-split results to match
    the paper's methodology; 5-fold cross-validation results are also
    available via evaluate_with_cv() for statistical context.
"""

import time
import logging
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
import lightgbm as lgb
from typing import Dict, Any, Tuple, Optional

logger = logging.getLogger(__name__)


# Paper-reported accuracy values (Table 2, Section 3)
PAPER_ACCURACIES = {
    'DT':   0.84,
    'LR':   0.85,
    'LGBM': 0.91,
    'SVC':  0.86,
}


def create_classifiers(random_seed: int = 42) -> Dict[str, Any]:
    """
    Create the four classifiers with parameters as specified by the paper.

    See module docstring for documented deviations from strict defaults.

    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Mapping abbreviation -> (name, model) tuples.
    """
    classifiers = {
        'DT': ('Decision Tree', DecisionTreeClassifier(random_state=random_seed)),
        'LR': ('Logistic Regression', LogisticRegression(
            max_iter=1000,          # deviation: default=100 (insufficient)
            random_state=random_seed,
        )),
        'LGBM': ('LightGBM', lgb.LGBMClassifier(
            random_state=random_seed,
            verbose=-1,
        )),
        'SVC': ('Support Vector Classifier', SVC(
            probability=True,       # deviation: required for LIME / ROC-AUC
            random_state=random_seed,
        )),
    }
    return classifiers


def train_model(model, X_train, y_train) -> Tuple[Any, float]:
    """
    Train a classifier and measure training time.

    Parameters
    ----------
    model : sklearn-compatible classifier
    X_train : array-like
    y_train : array-like

    Returns
    -------
    Tuple[model, float]
        Trained model and training time in seconds.
    """
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    return model, train_time


def evaluate_model(model, X_test, y_test, model_name: str = '') -> Dict[str, Any]:
    """
    Evaluate a trained classifier comprehensively.

    Parameters
    ----------
    model : trained classifier
    X_test : array-like
    y_test : array-like
    model_name : str

    Returns
    -------
    dict
        Evaluation metrics including accuracy, precision, recall, F1, ROC-AUC.
    """
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start

    metrics: Dict[str, Any] = {
        'model_name': model_name,
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, average='binary', zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, average='binary', zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred, average='binary', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'inference_time_seconds': inference_time,
    }

    # ROC-AUC — only catch the specific exceptions that are expected
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None

        if y_proba is not None:
            metrics['roc_auc'] = float(roc_auc_score(y_test, y_proba))
    except ValueError as e:
        # ValueError is raised when only one class is present in y_test
        metrics['roc_auc'] = None
        metrics['roc_auc_error'] = str(e)
        logger.warning("ROC-AUC could not be computed for %s: %s", model_name, e)

    return metrics


def evaluate_with_cv(model, X_train: pd.DataFrame, y_train: np.ndarray,
                     n_splits: int = 5, random_seed: int = 42) -> Dict[str, Any]:
    """
    Evaluate a model with stratified k-fold cross-validation on the training set.

    This provides more statistically reliable performance estimates than a
    single 80/20 split, especially with small datasets (~500 samples).

    Parameters
    ----------
    model : sklearn-compatible classifier
    X_train : pd.DataFrame
    y_train : np.ndarray
    n_splits : int
        Number of CV folds (default: 5).
    random_seed : int

    Returns
    -------
    dict
        Mean and std of accuracy, precision, recall, and F1 across folds.
    """
    from sklearn.base import clone

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    scoring = ['accuracy', 'precision', 'recall', 'f1']

    cv_results = cross_validate(
        clone(model), X_train, y_train,
        cv=cv, scoring=scoring, return_train_score=False,
    )

    summary: Dict[str, Any] = {'n_splits': n_splits}
    for metric in scoring:
        vals = cv_results[f'test_{metric}']
        summary[f'{metric}_mean'] = float(np.mean(vals))
        summary[f'{metric}_std'] = float(np.std(vals))
        summary[f'{metric}_95ci'] = float(1.96 * np.std(vals) / np.sqrt(n_splits))

    return summary


def train_and_evaluate_all(X_train, X_test, y_train, y_test,
                           random_seed: int = 42,
                           run_cv: bool = True) -> Dict[str, Dict]:
    """
    Train and evaluate all four classifiers.

    Parameters
    ----------
    X_train, X_test : array-like
    y_train, y_test : array-like
    random_seed : int
    run_cv : bool
        If True, also run 5-fold CV on the training set for statistical context.

    Returns
    -------
    dict
        Results for each model including trained model, metrics, and times.
    """
    classifiers = create_classifiers(random_seed)
    results: Dict[str, Dict] = {}

    for abbrev, (name, model) in classifiers.items():
        print(f"\n{'=' * 60}")
        print(f"Training {name} ({abbrev})...")
        print(f"{'=' * 60}")

        # Train
        trained_model, train_time = train_model(model, X_train, y_train)

        # Single-split evaluation
        metrics = evaluate_model(trained_model, X_test, y_test, name)
        metrics['train_time_seconds'] = train_time

        # Cross-validation for statistical reliability
        cv_metrics: Optional[Dict] = None
        if run_cv:
            print(f"  Running 5-fold CV for {abbrev}...")
            cv_metrics = evaluate_with_cv(
                trained_model, X_train, y_train, random_seed=random_seed
            )
            metrics['cv'] = cv_metrics

        # Compare with paper
        paper_acc = PAPER_ACCURACIES.get(abbrev)
        if paper_acc is not None:
            metrics['paper_accuracy'] = paper_acc
            metrics['accuracy_difference'] = abs(metrics['accuracy'] - paper_acc)

        results[abbrev] = {
            'name': name,
            'model': trained_model,
            'metrics': metrics,
        }

        print(f"  Accuracy:  {metrics['accuracy']:.4f} "
              f"(paper: {paper_acc:.2f}, "
              f"diff: {metrics.get('accuracy_difference', 'N/A'):.4f})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Train time: {train_time:.4f}s")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
        if cv_metrics:
            print(f"  CV Accuracy: {cv_metrics['accuracy_mean']:.4f} "
                  f"± {cv_metrics['accuracy_std']:.4f} "
                  f"(95% CI ±{cv_metrics['accuracy_95ci']:.4f})")

    return results


def create_performance_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create a comparison table of classifier performance.

    Parameters
    ----------
    results : dict
        Results from train_and_evaluate_all().

    Returns
    -------
    pd.DataFrame
        Performance comparison table.
    """
    rows = []
    for abbrev, res in results.items():
        m = res['metrics']
        cv = m.get('cv', {})
        rows.append({
            'Model': f"{res['name']} ({abbrev})",
            'Accuracy': m['accuracy'],
            'Paper Accuracy': m.get('paper_accuracy', None),
            'Abs Difference': m.get('accuracy_difference', None),
            'CV Accuracy (mean)': cv.get('accuracy_mean', None),
            'CV Accuracy (±std)': cv.get('accuracy_std', None),
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1-Score': m['f1_score'],
            'ROC-AUC': m.get('roc_auc', None),
            'Train Time (s)': m.get('train_time_seconds', None),
        })

    return pd.DataFrame(rows)
