# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Kubeflow Pipeline for training GNN+XGBoost fraud detection model."""

from kfp import compiler, dsl
from kfp import kubernetes

from .components.train_model import (
    prepare_training_config,
    train_fraud_model,
    upload_model_to_s3,
)

# ConfigMap name containing S3 configuration
MODEL_CONFIG_MAP = "fraud-detection-config"


@dsl.pipeline(
    name="fraud-detection-training-pipeline",
    description="Train GNN+XGBoost fraud detection model using NVIDIA container "
    "and upload to S3 for Triton deployment",
)
def fraud_detection_training_pipeline(
    # Data location (S3 URI to GNN preprocessed data)
    gnn_data_s3_uri: str = "s3://ml-on-containers/preprocessing/gnn",
    # GNN hyperparameters
    gnn_hidden_channels: int = 32,
    gnn_n_hops: int = 2,
    gnn_layer: str = "SAGEConv",
    gnn_dropout_prob: float = 0.1,
    gnn_batch_size: int = 4096,
    gnn_fan_out: int = 10,
    gnn_num_epochs: int = 8,
    # XGBoost hyperparameters
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.2,
    xgb_num_parallel_tree: int = 3,
    xgb_num_boost_round: int = 512,
    xgb_gamma: float = 0.0,
    # S3 configuration
    s3_model_prefix: str = "model-repository",
    s3_region: str = "us-east-1",
):
    """Train fraud detection model and upload to S3.

    This pipeline:
    1. Prepares training configuration with hyperparameters
    2. Trains a GNN+XGBoost model using NVIDIA's training container
    3. Uploads the trained model to S3 for Triton deployment

    S3 bucket for model storage is loaded from ConfigMap 'fraud-detection-config':
        - model_bucket: S3 bucket for model storage

    The pipeline requires GPU nodes (g4dn) for training.

    Args:
        gnn_data_s3_uri: S3 URI to preprocessed GNN data from preprocessing pipeline
        gnn_*: GNN hyperparameters
        xgb_*: XGBoost hyperparameters
        s3_model_prefix: S3 key prefix for model storage
        s3_region: AWS region for S3
    """
    # Step 1: Prepare training configuration
    config_task = prepare_training_config(
        gnn_hidden_channels=gnn_hidden_channels,
        gnn_n_hops=gnn_n_hops,
        gnn_layer=gnn_layer,
        gnn_dropout_prob=gnn_dropout_prob,
        gnn_batch_size=gnn_batch_size,
        gnn_fan_out=gnn_fan_out,
        gnn_num_epochs=gnn_num_epochs,
        xgb_max_depth=xgb_max_depth,
        xgb_learning_rate=xgb_learning_rate,
        xgb_num_parallel_tree=xgb_num_parallel_tree,
        xgb_num_boost_round=xgb_num_boost_round,
        xgb_gamma=xgb_gamma,
    )

    # Step 2: Train model using NVIDIA container
    train_task = train_fraud_model(
        gnn_data_uri=gnn_data_s3_uri,
        training_config=config_task.outputs["config_artifact"],
    )

    # Configure GPU node selector and tolerations for training
    kubernetes.add_node_selector(
        train_task,
        label_key="nvidia.com/gpu",
        label_value="true",
    )
    kubernetes.add_toleration(
        train_task,
        key="nvidia.com/gpu",
        operator="Exists",
        effect="NoSchedule",
    )

    # Request GPU resources
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)

    # Step 3: Upload model to S3
    upload_task = upload_model_to_s3(
        trained_model=train_task.outputs["trained_model"],
        s3_prefix=s3_model_prefix,
        s3_region=s3_region,
    )

    # Inject S3 bucket from ConfigMap
    kubernetes.use_config_map_as_env(
        upload_task,
        config_map_name=MODEL_CONFIG_MAP,
        config_map_key_to_env={
            "model_bucket": "MODEL_BUCKET",
        },
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_training_pipeline,
        package_path="fraud_detection_training_pipeline.yaml",
    )
    print("Pipeline compiled to fraud_detection_training_pipeline.yaml")
