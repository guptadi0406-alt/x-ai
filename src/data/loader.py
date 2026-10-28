"""
Dataset loading and validation for the Airline Passenger Satisfaction dataset.

The paper uses 500 subjects (250 satisfied) from the Kaggle Airline Passenger
Satisfaction dataset with 22 features for binary classification.

Original dataset: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path


# The 22 features used in the paper's case study (Section 3, Table 2)
PAPER_FEATURES = [
    'Inflight wifi service',
    'Type of Travel',
    'Online boarding',
    'Class',
    'Cleanliness',
    'On-board service',
    'Departure/Arrival time convenient',
    'Baggage handling',
    'Leg room service',
    'Food and drink',
    'Age',
    'Customer Type',
    'Flight Distance',
    'Arrival Delay in Minutes',
    'Ease of Online booking',
    'Seat comfort',
    'Gate location',
    'Inflight service',
    'Check-in service',
    'Departure Delay in Minutes',
    'Gender',
    'Inflight entertainment',
]

TARGET_COLUMN = 'satisfaction'
TARGET_POSITIVE = 'satisfied'
TARGET_NEGATIVE = 'neutral or dissatisfied'


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """
    Load the Airline Passenger Satisfaction dataset.
    
    The dataset should be a CSV file. If data_path is None, looks in 
    the default data/ directory.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the CSV file. If None, searches default locations.
        
    Returns
    -------
    pd.DataFrame
        The loaded dataset.
        
    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.
    """
    if data_path is None:
        # Search default locations
        project_root = Path(__file__).parent.parent.parent
        search_paths = [
            project_root / 'data' / 'airline_satisfaction.csv',
            project_root / 'data' / 'train.csv',
            project_root / 'data' / 'test.csv',
            project_root / 'data' / 'Invistico_Airline.csv',
        ]
        for path in search_paths:
            if path.exists():
                data_path = str(path)
                break
    
    if data_path is None or not os.path.exists(data_path):
        raise FileNotFoundError(
            "Dataset not found. Please download the Airline Passenger Satisfaction "
            "dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction\n"
            "Place the CSV file in the data/ directory.\n"
            "Alternatively, run: python -m src.data.download_dataset"
        )
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset from {data_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate that the dataset has the expected schema and report statistics.
    
    Parameters
    ----------
    df : pd.DataFrame
        The loaded dataset.
        
    Returns
    -------
    dict
        Validation results including column checks and statistics.
    """
    results = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': df.duplicated().sum(),
    }
    
    # Check for target column (case-insensitive matching)
    target_found = None
    for col in df.columns:
        if col.lower() == 'satisfaction':
            target_found = col
            break
    
    results['target_column'] = target_found
    if target_found:
        results['target_distribution'] = df[target_found].value_counts().to_dict()
    
    # Check for expected features
    found_features = []
    missing_features = []
    col_lower_map = {c.lower(): c for c in df.columns}
    
    for feat in PAPER_FEATURES:
        feat_lower = feat.lower()
        if feat_lower in col_lower_map:
            found_features.append(col_lower_map[feat_lower])
        else:
            # Try partial match
            matched = False
            for col_l, col_orig in col_lower_map.items():
                if feat_lower.replace(' ', '') in col_l.replace(' ', ''):
                    found_features.append(col_orig)
                    matched = True
                    break
            if not matched:
                missing_features.append(feat)
    
    results['found_features'] = found_features
    results['missing_features'] = missing_features
    results['n_features_found'] = len(found_features)
    
    return results


def sample_dataset(df: pd.DataFrame, n_samples: int = 500, 
                   n_positive: int = 250, random_seed: int = 42) -> pd.DataFrame:
    """
    Sample the dataset to match the paper's case study (500 subjects, 250 satisfied).
    
    The paper reports using 500 subjects with 250 satisfied. Since the original
    Kaggle dataset has ~130K observations, we perform stratified sampling.
    
    LIMITATION: The paper does not disclose the exact sampling procedure or
    random seed. This is our best approximation.
    
    Parameters
    ----------
    df : pd.DataFrame
        The full dataset.
    n_samples : int
        Total number of samples (default: 500 per paper).
    n_positive : int
        Number of satisfied passengers (default: 250 per paper).
    random_seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    pd.DataFrame
        The sampled dataset.
    """
    np.random.seed(random_seed)
    
    # Identify target column
    target_col = None
    for col in df.columns:
        if col.lower() == 'satisfaction':
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Cannot find 'satisfaction' column in dataset")
    
    # Separate by class
    satisfied = df[df[target_col].str.lower().str.contains('satisfied') & 
                   ~df[target_col].str.lower().str.contains('neutral|dissatisfied')]
    not_satisfied = df[~df.index.isin(satisfied.index)]
    
    n_negative = n_samples - n_positive
    
    if len(satisfied) < n_positive:
        raise ValueError(
            f"Not enough satisfied samples: need {n_positive}, have {len(satisfied)}"
        )
    if len(not_satisfied) < n_negative:
        raise ValueError(
            f"Not enough non-satisfied samples: need {n_negative}, have {len(not_satisfied)}"
        )
    
    # Sample
    satisfied_sample = satisfied.sample(n=n_positive, random_state=random_seed)
    not_satisfied_sample = not_satisfied.sample(n=n_negative, random_state=random_seed)
    
    sampled = pd.concat([satisfied_sample, not_satisfied_sample])
    sampled = sampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"Sampled {len(sampled)} observations: "
          f"{n_positive} satisfied, {n_negative} not satisfied")
    
    return sampled


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Identify the 22 feature columns in the dataset, matching the paper's feature list.
    
    Handles minor naming differences between dataset versions.
    
    Parameters
    ----------
    df : pd.DataFrame
        The dataset.
        
    Returns
    -------
    list
        List of column names matching the paper's 22 features.
    """
    col_lower_map = {c.lower().replace('_', ' ').strip(): c for c in df.columns}
    
    # Feature name mapping (paper name -> possible dataset names)
    feature_aliases = {
        'Inflight wifi service': ['inflight wifi service'],
        'Type of Travel': ['type of travel'],
        'Online boarding': ['online boarding'],
        'Class': ['class'],
        'Cleanliness': ['cleanliness'],
        'On-board service': ['on-board service', 'onboard service'],
        'Departure/Arrival time convenient': [
            'departure/arrival time convenient',
            'departure arrival time convenient'
        ],
        'Baggage handling': ['baggage handling'],
        'Leg room service': ['leg room service', 'legroom service'],
        'Food and drink': ['food and drink'],
        'Age': ['age'],
        'Customer Type': ['customer type'],
        'Flight Distance': ['flight distance'],
        'Arrival Delay in Minutes': ['arrival delay in minutes'],
        'Ease of Online booking': ['ease of online booking'],
        'Seat comfort': ['seat comfort'],
        'Gate location': ['gate location'],
        'Inflight service': ['inflight service'],
        'Check-in service': ['check-in service', 'checkin service'],
        'Departure Delay in Minutes': ['departure delay in minutes'],
        'Gender': ['gender'],
        'Inflight entertainment': ['inflight entertainment'],
    }
    
    matched_columns = []
    for paper_name, aliases in feature_aliases.items():
        found = False
        for alias in aliases:
            alias_clean = alias.lower().replace('_', ' ').strip()
            if alias_clean in col_lower_map:
                matched_columns.append(col_lower_map[alias_clean])
                found = True
                break
        if not found:
            # Try fuzzy match
            for col_clean, col_orig in col_lower_map.items():
                if paper_name.lower().replace(' ', '') in col_clean.replace(' ', ''):
                    matched_columns.append(col_orig)
                    found = True
                    break
        if not found:
            print(f"WARNING: Feature '{paper_name}' not found in dataset")
    
    print(f"Matched {len(matched_columns)}/{len(feature_aliases)} paper features")
    return matched_columns
