# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Kubeflow Pipeline for training GNN+XGBoost fraud detection model."""

from kfp import compiler, dsl
from kfp import kubernetes

from .components.train_model import (
    prepare_training_config,
    download_gnn_data_to_pvc,
    copy_config_to_pvc,
    run_nvidia_training,
    upload_model_from_pvc,
)
from .components.test_model import smoke_test_triton

# ConfigMap name containing S3 configuration
MODEL_CONFIG_MAP = "fraud-detection-config"

# PVC configuration
DATA_PVC_SIZE = "50Gi"  # GNN data can be large
OUTPUT_PVC_SIZE = "10Gi"  # Model output is smaller
CONFIG_PVC_SIZE = "1Gi"  # Config is tiny
STORAGE_CLASS = "gp3"  # AWS EBS gp3, adjust for your cluster


@dsl.pipeline(
    name="fraud-detection-training-pipeline",
    description="Train GNN+XGBoost fraud detection model using NVIDIA container "
    "and upload to S3 for Triton deployment. Uses PVCs for data transfer.",
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
    # Validation configuration
    run_smoke_test: bool = True,
    triton_service_name: str = "triton-inference-server",
    triton_namespace: str = "triton",
    triton_port: int = 8005,
):
    """Train fraud detection model and upload to S3.

    This pipeline:
    1. Creates PVCs for data transfer between components
    2. Downloads GNN data from S3 to data PVC
    3. Prepares training config and copies to config PVC
    4. Runs NVIDIA training container with PVCs mounted
    5. Uploads trained model from output PVC to S3
    6. Cleans up PVCs
    7. Optionally runs smoke test to verify Triton picks up new model

    S3 bucket for model storage is loaded from ConfigMap 'fraud-detection-config':
        - model_bucket: S3 bucket for model storage

    The pipeline requires GPU nodes (g4dn/p3/etc) for training.

    Args:
        gnn_data_s3_uri: S3 URI to preprocessed GNN data from preprocessing pipeline
        gnn_*: GNN hyperparameters
        xgb_*: XGBoost hyperparameters
        s3_model_prefix: S3 key prefix for model storage
        s3_region: AWS region for S3
        run_smoke_test: Whether to run smoke test after upload
        triton_service_name: Triton k8s service name
        triton_namespace: Namespace where Triton is deployed
        triton_port: Triton HTTP port
    """
    # =========================================================================
    # Step 1: Create PVCs for data transfer
    # =========================================================================

    # PVC for GNN input data
    data_pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-gnn-data",
        access_modes=["ReadWriteOnce"],
        size=DATA_PVC_SIZE,
        storage_class_name=STORAGE_CLASS,
    )

    # PVC for trained model output
    output_pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-model-output",
        access_modes=["ReadWriteOnce"],
        size=OUTPUT_PVC_SIZE,
        storage_class_name=STORAGE_CLASS,
    )

    # PVC for config file
    config_pvc = kubernetes.CreatePVC(
        pvc_name_suffix="-training-config",
        access_modes=["ReadWriteOnce"],
        size=CONFIG_PVC_SIZE,
        storage_class_name=STORAGE_CLASS,
    )

    # =========================================================================
    # Step 2: Prepare training config
    # =========================================================================

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

    # =========================================================================
    # Step 3: Download GNN data to PVC
    # =========================================================================

    download_task = download_gnn_data_to_pvc(
        gnn_data_s3_uri=gnn_data_s3_uri,
        s3_region=s3_region,
        mount_path="/data",
    )
    download_task.after(data_pvc)

    # Mount data PVC
    kubernetes.mount_pvc(
        download_task,
        pvc_name=data_pvc.outputs["name"],
        mount_path="/data",
    )

    # =========================================================================
    # Step 4: Copy config to PVC
    # =========================================================================

    copy_config_task = copy_config_to_pvc(
        training_config=config_task.outputs["config_artifact"],
        config_mount_path="/config",
    )
    copy_config_task.after(config_pvc)

    # Mount config PVC
    kubernetes.mount_pvc(
        copy_config_task,
        pvc_name=config_pvc.outputs["name"],
        mount_path="/config",
    )

    # =========================================================================
    # Step 5: Run NVIDIA training container
    # =========================================================================

    train_task = run_nvidia_training(
        data_mount_path="/data",
        output_mount_path="/trained_models",
        config_mount_path="/config",
    )
    train_task.after(download_task)
    train_task.after(copy_config_task)
    train_task.after(output_pvc)

    # Mount all three PVCs
    kubernetes.mount_pvc(
        train_task,
        pvc_name=data_pvc.outputs["name"],
        mount_path="/data",
    )
    kubernetes.mount_pvc(
        train_task,
        pvc_name=output_pvc.outputs["name"],
        mount_path="/trained_models",
    )
    kubernetes.mount_pvc(
        train_task,
        pvc_name=config_pvc.outputs["name"],
        mount_path="/config",
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

    # =========================================================================
    # Step 6: Upload model to S3
    # =========================================================================

    upload_task = upload_model_from_pvc(
        s3_prefix=s3_model_prefix,
        s3_region=s3_region,
        output_mount_path="/trained_models",
    )
    upload_task.after(train_task)

    # Mount output PVC to read trained model
    kubernetes.mount_pvc(
        upload_task,
        pvc_name=output_pvc.outputs["name"],
        mount_path="/trained_models",
    )

    # Inject S3 bucket from ConfigMap
    kubernetes.use_config_map_as_env(
        upload_task,
        config_map_name=MODEL_CONFIG_MAP,
        config_map_key_to_env={
            "model_bucket": "MODEL_BUCKET",
        },
    )

    # =========================================================================
    # Step 7: Cleanup PVCs
    # =========================================================================

    delete_data_pvc = kubernetes.DeletePVC(
        pvc_name=data_pvc.outputs["name"]
    ).after(upload_task)

    delete_output_pvc = kubernetes.DeletePVC(
        pvc_name=output_pvc.outputs["name"]
    ).after(upload_task)

    delete_config_pvc = kubernetes.DeletePVC(
        pvc_name=config_pvc.outputs["name"]
    ).after(upload_task)

    # =========================================================================
    # Step 8: Optional smoke test
    # =========================================================================

    with dsl.If(run_smoke_test == True):  # noqa: E712
        triton_host = f"{triton_service_name}.{triton_namespace}.svc.cluster.local"

        smoke_task = smoke_test_triton(
            triton_host=triton_host,
            triton_port=triton_port,
            model_name="prediction_and_shapley",
            timeout_seconds=120,
        )
        smoke_task.after(upload_task)


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=fraud_detection_training_pipeline,
        package_path="fraud_detection_training_pipeline.yaml",
    )
    print("Pipeline compiled to fraud_detection_training_pipeline.yaml")
