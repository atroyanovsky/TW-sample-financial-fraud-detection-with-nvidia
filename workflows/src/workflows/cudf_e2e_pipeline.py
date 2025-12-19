# Copyright (c) 2025, Amazon Web Services, Inc.
"""End-to-end Kubeflow Pipeline with RAPIDS/cuDF preprocessing.

This pipeline uses NVIDIA RAPIDS for GPU-accelerated preprocessing:
- S3 Input: Raw TabFormer CSV data + preprocessing script
- PVC: All intermediate data (cleaned, split, transformed, GNN-ready)
- S3 Output: Final trained model for Triton

Pipeline steps:
1. Download raw data from S3 to PVC
2. Run cuDF preprocessing (RAPIDS container, GPU)
3. Write training config
4. Run GNN+XGBoost training (NVIDIA container, GPU)
5. Upload model to S3

Compatible with KFP 2.1.x and kfp-kubernetes 1.0.x
"""

from kfp import compiler, dsl, kubernetes

# PVC configuration
DATA_PVC_SIZE = "100Gi"
MODEL_PVC_SIZE = "10Gi"
STORAGE_CLASS = "gp3"

# Container images
RAPIDS_IMAGE = "rapidsai/base:25.12-cuda13-py3.12"
TRAINING_IMAGE = (
    "915948456033.dkr.ecr.us-west-2.amazonaws.com/nvidia-training-repo:latest"
)


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "boto3",
        "botocore",
        "s3transfer",
        "jmespath",
        "python-dateutil",
        "urllib3",
        "six",
        "requests",
    ],
)
def download_raw_data_to_pvc(
    s3_bucket: str,
    s3_region: str,
    raw_data_path: str,
    script_url: str,
    data_mount_path: str = "/data",
):
    """Download raw CSV from S3 and preprocessing script from GitHub to PVC."""
    import os

    import boto3
    import requests

    s3 = boto3.client("s3", region_name=s3_region)

    # Download raw data
    raw_dir = os.path.join(data_mount_path, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    local_csv = os.path.join(raw_dir, "card_transaction.v1.csv")
    print(f"Downloading s3://{s3_bucket}/{raw_data_path} to {local_csv}")
    s3.download_file(s3_bucket, raw_data_path, local_csv)
    print(f"Downloaded {os.path.getsize(local_csv)} bytes")

    # Set permissions for non-root containers (RAPIDS runs as UID 1000)
    os.chmod(data_mount_path, 0o777)
    os.chmod(raw_dir, 0o777)
    os.chmod(local_csv, 0o666)

    # Download preprocessing script from GitHub
    local_script = os.path.join(data_mount_path, "preprocess_tabformer.py")
    print(f"Downloading {script_url} to {local_script}")
    resp = requests.get(script_url)
    resp.raise_for_status()
    with open(local_script, "w") as f:
        f.write(resp.text)
    os.chmod(local_script, 0o755)
    print(f"Downloaded preprocessing script ({len(resp.text)} bytes)")


@dsl.container_component
def run_cudf_preprocessing():
    """Run cuDF-powered preprocessing on GPU using RAPIDS container.

    Expects:
        /data/raw/card_transaction.v1.csv
        /data/preprocess_tabformer.py

    Produces:
        /data/xgb/{training,validation,test}.csv
        /data/gnn/nodes/{user,merchant}.csv
        /data/gnn/edges/{user_to_merchant,user_to_merchant_attr,user_to_merchant_label}.csv
        /data/gnn/test_gnn/...
    """
    return dsl.ContainerSpec(
        image=RAPIDS_IMAGE,
        command=["/bin/bash", "-c"],
        args=[
            "pip install category_encoders scikit-learn && "
            "echo '=== Starting cuDF preprocessing ===' && "
            "python /data/preprocess_tabformer.py /data && "
            "echo '=== Preprocessing complete ===' && "
            "echo 'GNN structure:' && "
            "find /data/gnn -type f | head -20"
        ],
    )


@dsl.component(base_image="python:3.11")
def prepare_training_config(
    data_mount_path: str = "/data",
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
):
    """Write training config JSON to PVC."""
    import json
    import os

    config = {
        "paths": {"data_dir": "/data/gnn", "output_dir": "/trained_models"},
        "models": [
            {
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
                        "num_epochs": gnn_num_epochs,
                    },
                    "xgb": {
                        "max_depth": xgb_max_depth,
                        "learning_rate": xgb_learning_rate,
                        "num_parallel_tree": xgb_num_parallel_tree,
                        "num_boost_round": xgb_num_boost_round,
                        "gamma": xgb_gamma,
                    },
                },
            }
        ],
    }

    config_path = os.path.join(data_mount_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")


@dsl.container_component
def run_nvidia_training():
    """Run NVIDIA GNN+XGBoost training container."""
    return dsl.ContainerSpec(
        image=TRAINING_IMAGE,
        command=["/bin/bash", "-c"],
        args=[
            "ln -sf /data/config.json /app/config.json && "
            "cat /app/config.json && "
            "echo '=== GNN Data ===' && ls -la /data/gnn/ && "
            "cd /app && "
            "torchrun --standalone --nnodes=1 --nproc-per-node=1 main.py --config /app/config.json && "
            "echo '=== Training Complete ===' && ls -la /trained_models/"
        ],
    )


@dsl.component(
    base_image="python:3.11",
    packages_to_install=[
        "boto3",
        "botocore",
        "s3transfer",
        "jmespath",
        "python-dateutil",
        "urllib3",
        "six",
    ],
)
def upload_model_to_s3(
    model_bucket: str,
    model_mount_path: str = "/trained_models",
    s3_prefix: str = "model-repository",
    s3_region: str = "us-west-2",
) -> dict:
    """Upload trained model from PVC to S3."""
    import os
    from pathlib import Path

    import boto3

    s3 = boto3.client("s3", region_name=s3_region)
    model_repo = Path(model_mount_path) / "python_backend_model_repository"
    if not model_repo.exists():
        model_repo = Path(model_mount_path)

    uploaded = 0
    total_bytes = 0
    for root, dirs, files in os.walk(model_repo):
        for f in files:
            local_path = os.path.join(root, f)
            rel_path = os.path.relpath(local_path, model_repo)
            s3_key = f"{s3_prefix}/{rel_path}"
            size = os.path.getsize(local_path)
            print(f"Uploading {rel_path} ({size} bytes)")
            s3.upload_file(local_path, model_bucket, s3_key)
            uploaded += 1
            total_bytes += size

    s3_uri = f"s3://{model_bucket}/{s3_prefix}"
    print(f"\nUploaded {uploaded} files ({total_bytes} bytes) to {s3_uri}")
    return {"s3_uri": s3_uri, "files": uploaded, "bytes": total_bytes}


@dsl.pipeline(
    name="fraud-detection-cudf-pipeline",
    description="End-to-end fraud detection with RAPIDS/cuDF preprocessing. "
    "Uses GPU for both preprocessing (cuDF) and training (GNN+XGBoost).",
)
def fraud_detection_cudf_pipeline(
    # S3 configuration
    s3_bucket: str = "ml-on-containers-915948456033",
    model_bucket: str = "ml-on-containers-915948456033-model-registry",
    s3_region: str = "us-west-2",
    raw_data_path: str = "data/TabFormer/raw/card_transaction.v1.csv",
    script_url: str = "https://raw.githubusercontent.com/aws-samples/sample-financial-fraud-detection-with-nvidia/refs/heads/v2/workflows/src/workflows/components/preprocess_tabformer.py",
    s3_model_prefix: str = "model-repository",
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
):
    """End-to-end fraud detection pipeline with cuDF preprocessing.

    Data flow:
    1. S3 -> Download raw TabFormer CSV + preprocessing script to PVC
    2. GitHub -> Download preprocessing script to PVC
    3. PVC -> Write training config
    4. PVC -> Train GNN+XGBoost model (GPU)
    5. PVC -> Upload trained model to S3
    """
    # Create PVCs
    data_pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-fraud-data",
        access_modes=["ReadWriteOnce"],
        size=DATA_PVC_SIZE,
        storage_class_name=STORAGE_CLASS,
    )

    model_pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-fraud-model",
        access_modes=["ReadWriteOnce"],
        size=MODEL_PVC_SIZE,
        storage_class_name=STORAGE_CLASS,
    )

    # Step 1: Download raw data from S3 and script from GitHub
    download_task = download_raw_data_to_pvc(
        s3_bucket=s3_bucket,
        s3_region=s3_region,
        raw_data_path=raw_data_path,
        script_url=script_url,
        data_mount_path="/data",
    )
    download_task.after(data_pvc)
    download_task.set_caching_options(False)
    kubernetes.mount_pvc(
        download_task, pvc_name=data_pvc.outputs["name"], mount_path="/data"
    )

    # Step 2: Run cuDF preprocessing (GPU)
    preprocess_task = run_cudf_preprocessing()
    preprocess_task.after(download_task)
    kubernetes.mount_pvc(
        preprocess_task, pvc_name=data_pvc.outputs["name"], mount_path="/data"
    )
    # GPU node selection for preprocessing
    kubernetes.add_node_selector(
        preprocess_task, label_key="nvidia.com/gpu", label_value="true"
    )
    preprocess_task.set_memory_request("16Gi").set_memory_limit("50Gi")
    preprocess_task.set_cpu_request("4").set_cpu_limit("8")
    preprocess_task.set_accelerator_limit(1)
    preprocess_task.set_accelerator_type("nvidia.com/gpu")
    preprocess_task.set_caching_options(False)

    # Step 3: Write training config
    config_task = prepare_training_config(
        data_mount_path="/data",
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
    config_task.after(preprocess_task)
    kubernetes.mount_pvc(
        config_task, pvc_name=data_pvc.outputs["name"], mount_path="/data"
    )
    config_task.set_caching_options(False)

    # Step 4: Train model (GPU)
    train_task = run_nvidia_training()
    train_task.after(config_task)
    train_task.after(model_pvc)
    kubernetes.mount_pvc(
        train_task, pvc_name=data_pvc.outputs["name"], mount_path="/data"
    )
    kubernetes.mount_pvc(
        train_task, pvc_name=model_pvc.outputs["name"], mount_path="/trained_models"
    )
    kubernetes.add_node_selector(
        train_task, label_key="nvidia.com/gpu", label_value="true"
    )
    train_task.set_memory_request("16Gi").set_memory_limit("32Gi")
    train_task.set_cpu_request("4").set_cpu_limit("8")
    train_task.set_accelerator_limit(1)
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_caching_options(False)

    # Step 5: Upload model to S3
    upload_task = upload_model_to_s3(
        model_bucket=model_bucket,
        model_mount_path="/trained_models",
        s3_prefix=s3_model_prefix,
        s3_region=s3_region,
    )
    upload_task.after(train_task)
    kubernetes.mount_pvc(
        upload_task, pvc_name=model_pvc.outputs["name"], mount_path="/trained_models"
    )
    upload_task.set_caching_options(False)

    # Step 6: Cleanup PVCs
    kubernetes.DeletePVC(pvc_name=data_pvc.outputs["name"]).after(upload_task)
    kubernetes.DeletePVC(pvc_name=model_pvc.outputs["name"]).after(upload_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_cudf_pipeline,
        package_path="fraud_detection_cudf_pipeline.yaml",
    )
    print("Pipeline compiled to fraud_detection_cudf_pipeline.yaml")
