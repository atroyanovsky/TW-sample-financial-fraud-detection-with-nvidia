# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Component: Prepare XGBoost-ready datasets."""

from kfp import dsl
from kfp.dsl import Artifact, Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.0",
        "numpy==1.26.0",
        "pyarrow==15.0.0",
    ],
)
def prepare_xgb_datasets(
    train_data: Input[Dataset],
    validation_data: Input[Dataset],
    test_data: Input[Dataset],
    feature_transformer_artifact: Input[Artifact],
    id_transformer_artifact: Input[Artifact],
    xgb_train: Output[Dataset],
    xgb_validation: Output[Dataset],
    xgb_test: Output[Dataset],
    metrics: Output[Metrics],
) -> dict:
    """Transform split datasets for XGBoost training.

    Applies fitted transformers to produce feature matrices suitable for XGBoost.
    Output includes transformed features + ID encodings + fraud label.

    Args:
        train_data: Input training split artifact
        validation_data: Input validation split artifact
        test_data: Input test split artifact
        feature_transformer_artifact: Fitted feature transformer
        id_transformer_artifact: Fitted ID transformer
        xgb_train: Output XGBoost training dataset
        xgb_validation: Output XGBoost validation dataset
        xgb_test: Output XGBoost test dataset
        metrics: Output metrics artifact

    Returns:
        Dict with transformation statistics
    """
    import pickle

    import pandas as pd

    COL_FRAUD = "Fraud"

    # Load transformers
    with open(feature_transformer_artifact.path, "rb") as f:
        feature_config = pickle.load(f)

    with open(id_transformer_artifact.path, "rb") as f:
        id_config = pickle.load(f)

    transformer = feature_config["transformer"]
    output_columns = feature_config["output_columns"]
    type_mapping = feature_config["type_mapping"]
    predictor_columns = feature_config["predictor_columns"]

    id_transformer = id_config["transformer"]
    id_columns = id_config["columns"]
    merchant_user_cols = id_config["merchant_user_cols"]

    def transform_split(data_path: str, output_path: str, split_name: str) -> dict:
        """Transform a single data split."""
        df = pd.read_parquet(data_path)
        n_records = len(df)

        # Transform features
        transformed = transformer.transform(df[predictor_columns])
        result = pd.DataFrame(transformed, columns=output_columns)

        # Transform IDs
        id_transformed = id_transformer.transform(
            df[merchant_user_cols].astype("category")
        )
        id_df = pd.DataFrame(id_transformed, columns=id_columns)

        # Combine
        result = pd.concat([result, id_df], axis=1)
        result[COL_FRAUD] = df[COL_FRAUD].values

        # Apply type mapping
        for col, dtype in type_mapping.items():
            if col in result.columns:
                result[col] = result[col].astype(dtype)

        # Validate
        assert result.columns[-1] == COL_FRAUD, "Fraud column must be last"

        # Save
        result.to_csv(output_path, index=False)

        fraud_count = result[COL_FRAUD].sum()
        print(
            f"{split_name}: {n_records:,} records, {fraud_count:,} fraud, {result.shape[1]} features"
        )

        return {
            "records": n_records,
            "fraud": int(fraud_count),
            "features": result.shape[1],
        }

    # Transform each split
    train_stats = transform_split(train_data.path, xgb_train.path, "Train")
    val_stats = transform_split(validation_data.path, xgb_validation.path, "Validation")
    test_stats = transform_split(test_data.path, xgb_test.path, "Test")

    # Log metrics
    metrics.log_metric("train_records", train_stats["records"])
    metrics.log_metric("validation_records", val_stats["records"])
    metrics.log_metric("test_records", test_stats["records"])
    metrics.log_metric("num_features", train_stats["features"])

    return {
        "train": train_stats,
        "validation": val_stats,
        "test": test_stats,
        "output_columns": output_columns + id_columns + [COL_FRAUD],
    }
