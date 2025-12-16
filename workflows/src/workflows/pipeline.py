# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Kubeflow Pipeline definition for TabFormer fraud detection preprocessing."""

from kfp import compiler, dsl, kubernetes

from .components.clean_data import clean_and_encode_data
from .components.fit_transformers import fit_transformers
from .components.load_data import load_raw_data
from .components.prepare_gnn import prepare_gnn_datasets
from .components.prepare_xgb import prepare_xgb_datasets
from .components.split_data import split_by_year
from .components.visualize import visualize_data_stats, visualize_graph_structure

# ConfigMap name containing data paths
DATA_CONFIG_MAP = "fraud-detection-config"


@dsl.pipeline(
    name="tabformer-preprocessing-pipeline",
    description="End-to-end data preprocessing for TabFormer fraud detection: "
    "load -> clean -> split -> transform -> prepare XGB/GNN datasets -> visualize",
)
def cc_data_preprocessing_pipeline(
    s3_region: str = "us-east-1",
    under_sample: bool = True,
    fraud_ratio: float = 0.1,
    train_year_cutoff: int = 2018,
    validation_year: int = 2018,
    one_hot_threshold: int = 8,
):
    """TabFormer preprocessing pipeline.

    Orchestrates the full preprocessing workflow:
    1. Load raw CSV data from S3 (path from ConfigMap)
    2. Clean and encode identifiers
    3. Split by year into train/validation/test
    4. Fit feature transformers on training data
    5. Prepare XGBoost-ready datasets
    6. Prepare GNN graph structures
    7. Generate visualizations for KFP UI

    Data paths are loaded from ConfigMap 'fraud-detection-config':
        - source_path: S3 key for raw CSV
        - s3_bucket: S3 bucket name

    Args:
        s3_region: AWS region for S3
        under_sample: Whether to undersample majority class
        fraud_ratio: Target fraud ratio when undersampling
        train_year_cutoff: Year cutoff for training data
        validation_year: Year for validation data
        one_hot_threshold: Max categories for one-hot (else binary)
    """
    # Step 1: Load raw data (paths injected from ConfigMap)
    load_task = load_raw_data(
        s3_region=s3_region,
    )
    kubernetes.use_config_map_as_env(
        load_task,
        config_map_name=DATA_CONFIG_MAP,
        config_map_key_to_env={
            "source_path": "SOURCE_PATH",
            "s3_bucket": "S3_BUCKET",
        },
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

    # Step 7: Generate visualizations for KFP UI
    # Data statistics visualization (runs after clean)
    visualize_data_stats(
        cleaned_data=clean_task.outputs["cleaned_data"],
    )

    # Graph structure visualization (runs after GNN prep, uses edge data)
    visualize_graph_structure(
        gnn_train_data=gnn_task.outputs["gnn_train_edges"],
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=cc_data_preprocessing_pipeline,
        package_path="cc_data_preprocessing_pipeline.yaml",
    )
    print("Pipeline compiled to cc_data_preprocessing_pipeline.yaml")
