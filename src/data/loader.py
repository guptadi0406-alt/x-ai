"""
Dataset loading and validation for the Airline Passenger Satisfaction dataset.

The paper uses 500 subjects (250 satisfied) from the Kaggle Airline Passenger
Satisfaction dataset with 22 features for binary classification.

Original dataset: https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction
"""

import logging
import os
import pandas as pd
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


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

    Parameters
    ----------
    data_path : str, optional
        Path to the CSV file. If None, searches default locations.

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If the dataset cannot be found.
    """
    if data_path is None:
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

    if data_path is None or not os.path.exists(str(data_path)):
        raise FileNotFoundError(
            "Dataset not found. Please download the Airline Passenger Satisfaction "
            "dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction\n"
            "Place the CSV file in the data/ directory."
        )

    df = pd.read_csv(data_path)
    logger.info("Loaded dataset from %s: %d rows, %d columns",
                data_path, df.shape[0], df.shape[1])
    print(f"Loaded dataset from {data_path}: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


def validate_dataset(df: pd.DataFrame) -> dict:
    """
    Validate that the dataset has the expected schema and report statistics.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    dict
        Validation results.
    """
    results = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'columns': list(df.columns),
        'missing_values': df.isnull().sum().to_dict(),
        'duplicates': int(df.duplicated().sum()),
    }

    # Case-insensitive target column detection
    target_found = None
    for col in df.columns:
        if col.lower() == 'satisfaction':
            target_found = col
            break

    results['target_column'] = target_found
    if target_found:
        # Exact-match filtering — more robust than str.contains()
        unique_vals = df[target_found].unique().tolist()
        results['target_unique_values'] = unique_vals
        results['target_distribution'] = df[target_found].value_counts().to_dict()

        n_classes = df[target_found].nunique()
        if n_classes != 2:
            logger.warning(
                "Target column has %d unique values (expected 2): %s",
                n_classes, unique_vals,
            )

    # Feature detection
    found_features = []
    missing_features = []
    col_lower_map = {c.lower(): c for c in df.columns}

    for feat in PAPER_FEATURES:
        feat_lower = feat.lower()
        if feat_lower in col_lower_map:
            found_features.append(col_lower_map[feat_lower])
        else:
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
    n_samples : int
        Total number of samples (default: 500 per paper).
    n_positive : int
        Number of satisfied passengers (default: 250 per paper).
    random_seed : int
        Random seed — passed directly to pd.DataFrame.sample(); does NOT
        mutate the global numpy random state (side-effect-free).

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    ValueError
        If the dataset contains an unexpected non-missing target label, or
        does not contain enough examples of either expected class.
    """
    # Identify target column
    target_col = None
    for col in df.columns:
        if col.lower() == 'satisfaction':
            target_col = col
            break

    if target_col is None:
        raise ValueError("Cannot find 'satisfaction' column in dataset")

    # Normalize labels, discard missing targets, and reject unexpected classes
    # rather than silently treating every non-positive value as a negative.
    labels = df[target_col].astype('string').str.strip().str.lower()
    missing_labels = labels.isna()
    if missing_labels.any():
        logger.warning(
            "Ignoring %d rows with missing satisfaction values",
            int(missing_labels.sum()),
        )

    unexpected_labels = labels[
        ~missing_labels & ~labels.isin([TARGET_POSITIVE, TARGET_NEGATIVE])
    ]
    if not unexpected_labels.empty:
        unexpected = sorted(unexpected_labels.unique().tolist())
        raise ValueError(
            "Unexpected values in 'satisfaction' column: "
            f"{unexpected}. Expected '{TARGET_POSITIVE}' or '{TARGET_NEGATIVE}'."
        )

    satisfied = df.loc[labels.eq(TARGET_POSITIVE).fillna(False)]
    not_satisfied = df.loc[labels.eq(TARGET_NEGATIVE).fillna(False)]

    n_negative = n_samples - n_positive

    if len(satisfied) < n_positive:
        raise ValueError(
            f"Not enough satisfied samples: need {n_positive}, have {len(satisfied)}"
        )
    if len(not_satisfied) < n_negative:
        raise ValueError(
            f"Not enough non-satisfied samples: need {n_negative}, have {len(not_satisfied)}"
        )

    # Use random_state parameter on .sample() — no global seed mutation
    satisfied_sample = satisfied.sample(n=n_positive, random_state=random_seed)
    not_satisfied_sample = not_satisfied.sample(n=n_negative, random_state=random_seed)

    sampled = pd.concat([satisfied_sample, not_satisfied_sample])
    sampled = sampled.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    logger.info("Sampled %d observations: %d satisfied, %d not satisfied",
                len(sampled), n_positive, n_negative)
    print(f"Sampled {len(sampled)} observations: "
          f"{n_positive} satisfied, {n_negative} not satisfied")

    return sampled


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Identify the 22 feature columns in the dataset, matching the paper's feature list.

    Handles minor naming differences between dataset versions using exact
    alias matching. Falls back to conservative fuzzy matching only when no
    alias matches (and deduplicates results).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    list
        Ordered list of column names matching the paper's 22 features.
    """
    col_lower_map = {c.lower().replace('_', ' ').strip(): c for c in df.columns}

    feature_aliases = {
        'Inflight wifi service':             ['inflight wifi service'],
        'Type of Travel':                    ['type of travel'],
        'Online boarding':                   ['online boarding'],
        'Class':                             ['class'],
        'Cleanliness':                       ['cleanliness'],
        'On-board service':                  ['on-board service', 'onboard service'],
        'Departure/Arrival time convenient': [
            'departure/arrival time convenient',
            'departure arrival time convenient',
        ],
        'Baggage handling':                  ['baggage handling'],
        'Leg room service':                  ['leg room service', 'legroom service'],
        'Food and drink':                    ['food and drink'],
        'Age':                               ['age'],
        'Customer Type':                     ['customer type'],
        'Flight Distance':                   ['flight distance'],
        'Arrival Delay in Minutes':          ['arrival delay in minutes'],
        'Ease of Online booking':            ['ease of online booking'],
        'Seat comfort':                      ['seat comfort'],
        'Gate location':                     ['gate location'],
        'Inflight service':                  ['inflight service'],
        'Check-in service':                  ['check-in service', 'checkin service'],
        'Departure Delay in Minutes':        ['departure delay in minutes'],
        'Gender':                            ['gender'],
        'Inflight entertainment':            ['inflight entertainment'],
    }

    matched_columns = []
    seen = set()  # Deduplicate in case multiple aliases hit the same column

    for paper_name, aliases in feature_aliases.items():
        found = False
        for alias in aliases:
            alias_clean = alias.lower().replace('_', ' ').strip()
            if alias_clean in col_lower_map:
                col = col_lower_map[alias_clean]
                if col not in seen:
                    matched_columns.append(col)
                    seen.add(col)
                found = True
                break

        if not found:
            logger.warning("Feature '%s' not found in dataset columns", paper_name)
            print(f"WARNING: Feature '{paper_name}' not found in dataset")

    logger.info("Matched %d/%d paper features", len(matched_columns), len(feature_aliases))
    print(f"Matched {len(matched_columns)}/{len(feature_aliases)} paper features")
    return matched_columns
