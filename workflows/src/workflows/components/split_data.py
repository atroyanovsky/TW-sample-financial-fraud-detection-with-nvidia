# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Component: Split data by year into train/validation/test sets."""

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas==2.2.0", "pyarrow==15.0.0"],
)
def split_by_year(
    cleaned_data: Input[Dataset],
    train_data: Output[Dataset],
    validation_data: Output[Dataset],
    test_data: Output[Dataset],
    metrics: Output[Metrics],
    train_year_cutoff: int = 2018,
    validation_year: int = 2018,
) -> dict:
    """Split cleaned data into train/validation/test by year.

    Default split:
    - Training: Year < 2018
    - Validation: Year == 2018
    - Test: Year > 2018

    Args:
        cleaned_data: Input cleaned dataset artifact
        train_data: Output training dataset artifact
        validation_data: Output validation dataset artifact
        test_data: Output test dataset artifact
        metrics: Output metrics artifact
        train_year_cutoff: Year cutoff for training data (exclusive)
        validation_year: Year for validation data

    Returns:
        Dict with split statistics
    """
    import pandas as pd

    COL_YEAR = "Year"
    COL_FRAUD = "Fraud"

    data = pd.read_parquet(cleaned_data.path)
    total_count = len(data)

    # Create split indices
    train_mask = data[COL_YEAR] < train_year_cutoff
    validation_mask = data[COL_YEAR] == validation_year
    test_mask = data[COL_YEAR] > validation_year

    train_df = data[train_mask].reset_index(drop=True)
    validation_df = data[validation_mask].reset_index(drop=True)
    test_df = data[test_mask].reset_index(drop=True)

    # Validate split covers all data
    assert len(train_df) + len(validation_df) + len(test_df) == total_count, (
        "Split does not account for all records"
    )

    # Calculate statistics
    stats = {
        "total_records": total_count,
        "train_records": len(train_df),
        "validation_records": len(validation_df),
        "test_records": len(test_df),
        "train_fraud_rate": float(train_df[COL_FRAUD].mean())
        if len(train_df) > 0
        else 0,
        "validation_fraud_rate": float(validation_df[COL_FRAUD].mean())
        if len(validation_df) > 0
        else 0,
        "test_fraud_rate": float(test_df[COL_FRAUD].mean()) if len(test_df) > 0 else 0,
    }

    print(
        f"Train: {stats['train_records']:,} ({100 * stats['train_records'] / total_count:.1f}%)"
    )
    print(
        f"Validation: {stats['validation_records']:,} ({100 * stats['validation_records'] / total_count:.1f}%)"
    )
    print(
        f"Test: {stats['test_records']:,} ({100 * stats['test_records'] / total_count:.1f}%)"
    )

    # Save splits
    train_df.to_parquet(train_data.path, index=False)
    validation_df.to_parquet(validation_data.path, index=False)
    test_df.to_parquet(test_data.path, index=False)

    # Log metrics
    metrics.log_metric("train_records", stats["train_records"])
    metrics.log_metric("validation_records", stats["validation_records"])
    metrics.log_metric("test_records", stats["test_records"])
    metrics.log_metric("train_fraud_rate", stats["train_fraud_rate"])
    metrics.log_metric("validation_fraud_rate", stats["validation_fraud_rate"])
    metrics.log_metric("test_fraud_rate", stats["test_fraud_rate"])

    return stats
