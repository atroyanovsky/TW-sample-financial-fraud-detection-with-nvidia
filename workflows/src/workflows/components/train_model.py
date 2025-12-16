# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Component: Train GNN+XGBoost fraud detection model using NVIDIA container."""

import json
from kfp import dsl
from kfp.dsl import Artifact, Input, Metrics, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["boto3==1.34.0"],
)
def prepare_training_config(
    config_artifact: Output[Artifact],
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
) -> dict:
    """Prepare training configuration JSON for NVIDIA training container.

    Args:
        config_artifact: Output artifact containing training config JSON
        gnn_*: GNN hyperparameters
        xgb_*: XGBoost hyperparameters

    Returns:
        Dict with config summary
    """
    import json

    config = {
        "paths": {
            "data_dir": "/data",
            "output_dir": "/trained_models"
        },
        "models": [{
            "kind": "GNN_XGBoost",
            "gpu": "single",
            "hyperparameters": {
                "gnn": {
                    "hidden_channels": gnn_hidden_channels,
                    "n_hops": gnn_n_hops,
                    "layer": gnn_layer,
                    "dropout_prob": gnn_dropout_prob,
                    "batch_size": gnn_batch_size,
                    "fan_out": gnn_fan_out,
                    "num_epochs": gnn_num_epochs
                },
                "xgb": {
                    "max_depth": xgb_max_depth,
                    "learning_rate": xgb_learning_rate,
                    "num_parallel_tree": xgb_num_parallel_tree,
                    "num_boost_round": xgb_num_boost_round,
                    "gamma": xgb_gamma
                }
            }
        }]
    }

    with open(config_artifact.path, 'w') as f:
        json.dump(config, f, indent=2)

    print("Training config:")
    print(json.dumps(config, indent=2))

    return {
        "gnn_epochs": gnn_num_epochs,
        "xgb_rounds": xgb_num_boost_round,
        "model_kind": "GNN_XGBoost",
    }


@dsl.container_component
def train_fraud_model(
    gnn_data_uri: str,
    training_config: Input[Artifact],
    trained_model: Output[Artifact],
):
    """Train GNN+XGBoost fraud detection model.

    Uses NVIDIA's financial-fraud-training container to train a hybrid
    GNN + XGBoost model for fraud detection.

    Args:
        gnn_data_uri: S3 URI to GNN data directory
        training_config: Training configuration JSON artifact
        trained_model: Output artifact for trained model repository

    Output structure:
        trained_model/
        └── python_backend_model_repository/
            └── prediction_and_shapley/
                ├── 1/
                │   ├── embedding_based_xgboost.json
                │   ├── model.py
                │   ├── meta.json
                │   └── state_dict_gnn_model.pth
                └── config.pbtxt
    """
    return dsl.ContainerSpec(
        image="nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0",
        command=["/bin/bash", "-c"],
        args=[
            # The container entrypoint handles training
            # We just need to ensure config is in place
            "cp {{$.inputs.artifacts['training_config'].path}} /app/config.json && "
            "python /app/main.py"
        ],
    )


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["boto3==1.34.0"],
)
def upload_model_to_s3(
    trained_model: Input[Artifact],
    model_s3_uri: Output[Artifact],
    s3_prefix: str = "model-repository",
    s3_region: str = "us-east-1",
) -> dict:
    """Upload trained model to S3.

    Uploads the trained model repository to S3 for deployment with Triton.
    S3 bucket is read from S3_BUCKET environment variable (injected via ConfigMap).

    Args:
        trained_model: Local trained model artifact
        model_s3_uri: Output artifact with S3 URI
        s3_prefix: S3 key prefix for model
        s3_region: AWS region

    Returns:
        Dict with S3 URI and upload stats
    """
    import os
    import boto3
    from pathlib import Path
    from datetime import datetime

    s3_bucket = os.environ.get("MODEL_BUCKET", "")
    if not s3_bucket:
        raise ValueError("MODEL_BUCKET environment variable not set")

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    s3 = boto3.client("s3", region_name=s3_region)
    model_path = Path(trained_model.path)

    # The training container outputs to python_backend_model_repository/
    # We need to upload the contents to s3_prefix/ (model-repository/)
    # Triton expects: s3://bucket/model-repository/prediction_and_shapley/...
    model_repo_path = model_path / "python_backend_model_repository"
    if not model_repo_path.exists():
        model_repo_path = model_path  # fallback if structure differs

    uploaded_files = 0
    total_bytes = 0

    # Upload all files in the model directory
    for root, dirs, files in os.walk(model_repo_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, model_repo_path)
            s3_key = f"{s3_prefix}/{relative_path}"

            file_size = os.path.getsize(local_path)
            print(f"Uploading {relative_path} ({file_size} bytes)")

            s3.upload_file(local_path, s3_bucket, s3_key)
            uploaded_files += 1
            total_bytes += file_size

    s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"Uploaded {uploaded_files} files ({total_bytes} bytes) to {s3_uri}")

    # Write S3 URI to output artifact
    with open(model_s3_uri.path, "w") as f:
        f.write(s3_uri)

    return {
        "s3_uri": s3_uri,
        "uploaded_files": uploaded_files,
        "total_bytes": total_bytes,
        "timestamp": timestamp,
    }
