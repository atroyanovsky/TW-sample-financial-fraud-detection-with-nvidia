# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Component: Train GNN+XGBoost fraud detection model using NVIDIA container."""

from kfp import dsl
from kfp.dsl import Artifact, Input, Output


@dsl.component(
    base_image="python:3.12",
)
def prepare_training_config(
    config_artifact: Output[Artifact],
    gnn_hidden_channels: int = 32,
    gnn_n_hops: int = 2,
    gnn_layer: str = "SAGEConv",
    gnn_dropout_prob: float = 0.1,
    gnn_batch_size: int = 4096,
    gnn_fan_out: int = 10,
    gnn_num_epochs: int = 8,
    xgb_max_depth: int = 6,
    xgb_learning_rate: float = 0.2,
    xgb_num_parallel_tree: int = 3,
    xgb_num_boost_round: int = 512,
    xgb_gamma: float = 0.0,
) -> dict:
    """Prepare training configuration JSON for NVIDIA training container.

    Creates the config.json that the NVIDIA financial-fraud-training container
    expects at /app/config.json. Paths are hardcoded to container mount points.

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


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["boto3==1.34.0"],
)
def download_gnn_data_to_pvc(
    gnn_data_s3_uri: str,
    s3_region: str = "us-east-1",
    mount_path: str = "/data",
):
    """Download GNN data from S3 to a mounted PVC.

    This component should have a PVC mounted at mount_path.
    It downloads all GNN preprocessing outputs from S3.

    Args:
        gnn_data_s3_uri: S3 URI (e.g., s3://bucket/preprocessing/gnn)
        s3_region: AWS region
        mount_path: Where PVC is mounted (default /data)
    """
    import os
    import boto3
    from urllib.parse import urlparse

    parsed = urlparse(gnn_data_s3_uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')

    s3 = boto3.client('s3', region_name=s3_region)

    print(f"Downloading from s3://{bucket}/{prefix} to {mount_path}")

    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            relative_path = key[len(prefix):].lstrip('/')
            local_path = os.path.join(mount_path, relative_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            print(f"  {key} -> {local_path}")
            s3.download_file(bucket, key, local_path)

    print(f"\nContents of {mount_path}:")
    for root, dirs, files in os.walk(mount_path):
        for f in files:
            path = os.path.join(root, f)
            size = os.path.getsize(path)
            print(f"  {os.path.relpath(path, mount_path)} ({size} bytes)")


@dsl.component(
    base_image="python:3.12",
)
def copy_config_to_pvc(
    training_config: Input[Artifact],
    config_mount_path: str = "/config",
):
    """Copy training config to a mounted PVC.

    The training container expects config at /app/config.json.
    We copy to PVC so it persists for the training container.

    Args:
        training_config: Training config artifact
        config_mount_path: Where config PVC is mounted
    """
    import shutil
    import os

    dest_path = os.path.join(config_mount_path, "config.json")
    shutil.copy(training_config.path, dest_path)

    print(f"Copied config to {dest_path}")
    with open(dest_path) as f:
        print(f.read())


@dsl.container_component
def run_nvidia_training(
    data_mount_path: str = "/data",
    output_mount_path: str = "/trained_models",
    config_mount_path: str = "/config",
):
    """Run NVIDIA training container with mounted PVCs.

    Expects three PVCs mounted:
    - data_mount_path: Contains GNN preprocessed data
    - output_mount_path: Where trained model will be written
    - config_mount_path: Contains config.json

    The container reads config from /app/config.json, so we symlink it.

    Args:
        data_mount_path: Mount point for GNN data PVC
        output_mount_path: Mount point for output PVC
        config_mount_path: Mount point for config PVC
    """
    return dsl.ContainerSpec(
        image="nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0",
        command=["/bin/bash", "-c"],
        args=[
            # Link config to expected location and run training
            f"ln -sf {config_mount_path}/config.json /app/config.json && "
            "cat /app/config.json && "
            f"echo '=== Data contents ===' && ls -la {data_mount_path}/ && "
            "echo '=== Starting training ===' && "
            "cd /app && python main.py && "
            f"echo '=== Training complete ===' && ls -la {output_mount_path}/"
        ],
    )


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["boto3==1.34.0"],
)
def upload_model_from_pvc(
    s3_prefix: str = "model-repository",
    s3_region: str = "us-east-1",
    output_mount_path: str = "/trained_models",
) -> dict:
    """Upload trained model from PVC to S3.

    Reads MODEL_BUCKET from environment (injected via ConfigMap).
    Uploads the model repository structure that Triton expects.

    The training container outputs:
        /trained_models/python_backend_model_repository/
        └── prediction_and_shapley/
            ├── 1/
            │   └── ... model files ...
            └── config.pbtxt

    We upload to s3://bucket/model-repository/prediction_and_shapley/...

    Args:
        s3_prefix: S3 key prefix (default: model-repository)
        s3_region: AWS region
        output_mount_path: Where output PVC is mounted

    Returns:
        Dict with S3 URI and upload stats
    """
    import os
    import boto3
    from pathlib import Path

    s3_bucket = os.environ.get("MODEL_BUCKET", "")
    if not s3_bucket:
        raise ValueError("MODEL_BUCKET environment variable not set")

    s3 = boto3.client("s3", region_name=s3_region)

    # Find the model repository in the output
    model_repo_path = Path(output_mount_path) / "python_backend_model_repository"
    if not model_repo_path.exists():
        model_repo_path = Path(output_mount_path)
        print(f"Warning: python_backend_model_repository not found, using {output_mount_path}")

    print(f"Model repo contents at {model_repo_path}:")
    for item in model_repo_path.rglob("*"):
        if item.is_file():
            print(f"  {item.relative_to(model_repo_path)}")

    uploaded_files = 0
    total_bytes = 0

    for root, dirs, files in os.walk(model_repo_path):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, model_repo_path)
            s3_key = f"{s3_prefix}/{relative_path}"

            file_size = os.path.getsize(local_path)
            print(f"Uploading {relative_path} ({file_size} bytes) -> s3://{s3_bucket}/{s3_key}")

            s3.upload_file(local_path, s3_bucket, s3_key)
            uploaded_files += 1
            total_bytes += file_size

    s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"\nUploaded {uploaded_files} files ({total_bytes} bytes) to {s3_uri}")

    return {
        "s3_uri": s3_uri,
        "uploaded_files": uploaded_files,
        "total_bytes": total_bytes,
    }
