# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Shared transformation utilities for TabFormer preprocessing."""

import numpy as np
import scipy.stats as ss


def cramers_v(x, y):
    """Compute Cramer's V correlation coefficient between categorical variables.

    See https://en.wikipedia.org/wiki/Cramér's_V

    Args:
        x: Categorical variable (pandas/cudf Series)
        y: Target variable (pandas/cudf Series)

    Returns:
        float: Cramer's V correlation coefficient
    """
    # Handle both pandas and cudf
    try:
        import cudf

        if isinstance(x, cudf.Series):
            confusion_matrix = cudf.crosstab(x, y).to_numpy()
        else:
            import pandas as pd

            confusion_matrix = pd.crosstab(x, y).to_numpy()
    except ImportError:
        import pandas as pd

        confusion_matrix = pd.crosstab(x, y).to_numpy()

    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r, k = confusion_matrix.shape
    return np.sqrt(chi2 / (n * (min(k - 1, r - 1))))


def create_feature_mask(
    columns: list[str], start_mask_id: int = 0
) -> tuple[dict, np.ndarray]:
    """Create a feature mask mapping columns to feature group IDs.

    For encoded columns (containing '_'), groups by the base feature name.
    This is used for GNN feature aggregation.

    Args:
        columns: List of column names
        start_mask_id: Starting ID for mask values

    Returns:
        Tuple of (mask_mapping dict, feature_mask array)
    """
    mask_mapping = {}
    mask_values = []
    current_mask = start_mask_id

    for col in columns:
        # For encoded columns, group by base feature name
        base_feature = col.split("_")[0] if "_" in col else col

        if base_feature not in mask_mapping:
            mask_mapping[base_feature] = current_mask
            current_mask += 1

        mask_values.append(mask_mapping[base_feature])

    return mask_mapping, np.array(mask_values)


def compute_correlations(
    data,
    target_col: str,
    categorical_cols: list[str],
    numerical_cols: list[str],
    sparse_factor: int = 1,
) -> dict:
    """Compute correlations between features and target.

    Args:
        data: DataFrame with features and target
        target_col: Name of target column
        categorical_cols: List of categorical column names
        numerical_cols: List of numerical column names
        sparse_factor: Subsampling factor for large datasets

    Returns:
        Dict with correlation results
    """
    from scipy.stats import pointbiserialr

    results = {"categorical": {}, "numerical": {}}

    for col in categorical_cols:
        coeff = cramers_v(data[col][::sparse_factor], data[target_col][::sparse_factor])
        results["categorical"][col] = coeff

    for col in numerical_cols:
        r_pb, p_value = pointbiserialr(data[target_col], data[col])
        results["numerical"][col] = {"r_pb": r_pb, "p_value": p_value}

    return results
