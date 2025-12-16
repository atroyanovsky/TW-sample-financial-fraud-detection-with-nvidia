# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""Kubeflow Pipeline definition for TabFormer fraud detection preprocessing."""

from kfp import dsl
from kfp import compiler

from components.load_data import load_raw_data
from components.clean_data import clean_and_encode_data
from components.split_data import split_by_year
from components.fit_transformers import fit_transformers
from components.prepare_xgb import prepare_xgb_datasets
from components.prepare_gnn import prepare_gnn_datasets


@dsl.pipeline(
    name="tabformer-preprocessing-pipeline",
    description="End-to-end data preprocessing for TabFormer fraud detection: "
                "load -> clean -> split -> transform -> prepare XGB/GNN datasets",
)
def tabformer_preprocessing_pipeline(
    source_path: str = "data/TabFormer/raw/card_transaction.v1.csv",
    s3_bucket: str = "",
    s3_region: str = "us-east-1",
    under_sample: bool = True,
    fraud_ratio: float = 0.1,
    train_year_cutoff: int = 2018,
    validation_year: int = 2018,
    one_hot_threshold: int = 8,
):
    """TabFormer preprocessing pipeline.

    Orchestrates the full preprocessing workflow:
    1. Load raw CSV data (from S3 or local)
    2. Clean and encode identifiers
    3. Split by year into train/validation/test
    4. Fit feature transformers on training data
    5. Prepare XGBoost-ready datasets
    6. Prepare GNN graph structures

    Args:
        source_path: Path to raw CSV (S3 key if bucket provided, else local)
        s3_bucket: Optional S3 bucket name for remote data
        s3_region: AWS region for S3
        under_sample: Whether to undersample majority class
        fraud_ratio: Target fraud ratio when undersampling
        train_year_cutoff: Year cutoff for training data
        validation_year: Year for validation data
        one_hot_threshold: Max categories for one-hot (else binary)
    """
    # Step 1: Load raw data
    load_task = load_raw_data(
        source_path=source_path,
        s3_bucket=s3_bucket,
        s3_region=s3_region,
    )

    # Step 2: Clean and encode
    clean_task = clean_and_encode_data(
        raw_data=load_task.outputs["raw_data"],
        under_sample=under_sample,
        fraud_ratio=fraud_ratio,
    )

    # Step 3: Split by year
    split_task = split_by_year(
        cleaned_data=clean_task.outputs["cleaned_data"],
        train_year_cutoff=train_year_cutoff,
        validation_year=validation_year,
    )

    # Step 4: Fit transformers on training data
    fit_task = fit_transformers(
        train_data=split_task.outputs["train_data"],
        one_hot_threshold=one_hot_threshold,
    )

    # Step 5: Prepare XGBoost datasets (parallel with GNN)
    xgb_task = prepare_xgb_datasets(
        train_data=split_task.outputs["train_data"],
        validation_data=split_task.outputs["validation_data"],
        test_data=split_task.outputs["test_data"],
        feature_transformer_artifact=fit_task.outputs["feature_transformer_artifact"],
        id_transformer_artifact=clean_task.outputs["id_transformer_artifact"],
    )

    # Step 6: Prepare GNN datasets (parallel with XGB)
    gnn_task = prepare_gnn_datasets(
        train_data=split_task.outputs["train_data"],
        validation_data=split_task.outputs["validation_data"],
        test_data=split_task.outputs["test_data"],
        feature_transformer_artifact=fit_task.outputs["feature_transformer_artifact"],
        id_transformer_artifact=clean_task.outputs["id_transformer_artifact"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=tabformer_preprocessing_pipeline,
        package_path="tabformer_preprocessing_pipeline.yaml",
    )
    print("Pipeline compiled to tabformer_preprocessing_pipeline.yaml")
