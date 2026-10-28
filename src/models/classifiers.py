"""
Classifier training and evaluation module.

Implements the four classifiers from the paper:
- Decision Tree (DT)
- Logistic Regression (LR)
- LightGBM (LGBM)
- Support Vector Classifier (SVC)

Paper reports using default parameters for all models.
"""

import time
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import lightgbm as lgb
from typing import Dict, Any, Tuple


# Paper-reported accuracy values (Table 2, Section 3)
PAPER_ACCURACIES = {
    'DT': 0.84,
    'LR': 0.85,
    'LGBM': 0.91,
    'SVC': 0.86,
}


def create_classifiers(random_seed: int = 42) -> Dict[str, Any]:
    """
    Create the four classifiers with default parameters as specified by the paper.
    
    The paper states: "The default parameters of each model were used."
    
    Parameters
    ----------
    random_seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    dict
        Dictionary mapping abbreviation to (name, model) tuples.
    """
    classifiers = {
        'DT': ('Decision Tree', DecisionTreeClassifier(random_state=random_seed)),
        'LR': ('Logistic Regression', LogisticRegression(
            max_iter=1000, random_state=random_seed
        )),
        'LGBM': ('LightGBM', lgb.LGBMClassifier(
            random_state=random_seed, verbose=-1
        )),
        'SVC': ('Support Vector Classifier', SVC(
            probability=True,  # needed for LIME and ROC-AUC
            random_state=random_seed
        )),
    }
    return classifiers


def train_model(model, X_train, y_train) -> Tuple[Any, float]:
    """
    Train a classifier and measure training time.
    
    Parameters
    ----------
    model : sklearn-compatible classifier
        The model to train.
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
        
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
        The model to evaluate.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    model_name : str
        Name for reporting.
        
    Returns
    -------
    dict
        Evaluation metrics.
    """
    start = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - start
    
    metrics = {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='binary', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='binary', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'inference_time_seconds': inference_time,
    }
    
    # ROC-AUC (requires probability estimates)
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = None
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
    except Exception as e:
        metrics['roc_auc'] = None
        metrics['roc_auc_error'] = str(e)
    
    return metrics


def train_and_evaluate_all(X_train, X_test, y_train, y_test,
                           random_seed: int = 42) -> Dict[str, Dict]:
    """
    Train and evaluate all four classifiers.
    
    Parameters
    ----------
    X_train, X_test : array-like
        Train and test features.
    y_train, y_test : array-like
        Train and test labels.
    random_seed : int
        Random seed.
        
    Returns
    -------
    dict
        Results for each model including trained model, metrics, and times.
    """
    classifiers = create_classifiers(random_seed)
    results = {}
    
    for abbrev, (name, model) in classifiers.items():
        print(f"\n{'='*60}")
        print(f"Training {name} ({abbrev})...")
        print(f"{'='*60}")
        
        # Train
        trained_model, train_time = train_model(model, X_train, y_train)
        
        # Evaluate
        metrics = evaluate_model(trained_model, X_test, y_test, name)
        metrics['train_time_seconds'] = train_time
        
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
              f"(paper: {paper_acc:.2f}, diff: {metrics.get('accuracy_difference', 'N/A'):.4f})")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if metrics.get('roc_auc') is not None:
            print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Train time: {train_time:.4f}s")
        print(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")
    
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
        rows.append({
            'Model': f"{res['name']} ({abbrev})",
            'Accuracy': m['accuracy'],
            'Paper Accuracy': m.get('paper_accuracy', None),
            'Abs Difference': m.get('accuracy_difference', None),
            'Precision': m['precision'],
            'Recall': m['recall'],
            'F1-Score': m['f1_score'],
            'ROC-AUC': m.get('roc_auc', None),
            'Train Time (s)': m.get('train_time_seconds', None),
        })
    
    return pd.DataFrame(rows)
