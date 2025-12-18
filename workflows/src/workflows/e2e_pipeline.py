# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""End-to-end Kubeflow Pipeline: Preprocessing + Training with PVC-based storage.

This pipeline minimizes S3 usage:
- S3 Input: Raw TabFormer CSV data
- PVC: All intermediate data (cleaned, split, transformed, GNN-ready)
- S3 Output: Final trained model for Triton
"""

from kfp import compiler, dsl
from kfp import kubernetes

from .components.test_model import smoke_test_triton

# ConfigMap name containing S3 configuration
CONFIG_MAP = "fraud-detection-config"

# PVC configuration
DATA_PVC_SIZE = "100Gi"  # Large enough for raw + processed data
MODEL_PVC_SIZE = "10Gi"  # Model output
STORAGE_CLASS = "gp3"  # AWS EBS gp3


# =============================================================================
# PVC-based preprocessing components
# =============================================================================


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["boto3==1.34.0", "pandas==2.2.0", "pyarrow==15.0.0"],
)
def load_raw_data_to_pvc(
    s3_region: str = "us-east-1",
    source_path: str = "data/TabFormer/raw/card_transaction.v1.csv",
    data_mount_path: str = "/data",
):
    """Load raw TabFormer CSV from S3 to PVC.

    Reads S3_BUCKET from environment (ConfigMap).
    Source path defaults to data/TabFormer/raw/card_transaction.v1.csv.

    Args:
        s3_region: AWS region
        source_path: S3 key for raw CSV file
        data_mount_path: Where data PVC is mounted
    """
    import os
    import boto3
    import pandas as pd

    s3_bucket = os.environ.get("S3_BUCKET", "")
    if not s3_bucket:
        raise ValueError("S3_BUCKET environment variable required")

    s3 = boto3.client("s3", region_name=s3_region)

    # Download raw CSV
    raw_dir = os.path.join(data_mount_path, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    local_path = os.path.join(raw_dir, "card_transaction.csv")

    print(f"Downloading s3://{s3_bucket}/{source_path} to {local_path}")
    s3.download_file(s3_bucket, source_path, local_path)

    # Also save as parquet for faster reads
    print("Converting to parquet...")
    df = pd.read_csv(local_path)
    parquet_path = os.path.join(raw_dir, "card_transaction.parquet")
    df.to_parquet(parquet_path, index=False)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {list(df.columns)}")


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "pandas==2.2.0",
        "numpy==1.26.0",
        "category_encoders==2.6.0",
        "scikit-learn==1.4.0",
        "pyarrow==15.0.0",
    ],
)
def preprocess_data_on_pvc(
    data_mount_path: str = "/data",
    under_sample: bool = True,
    fraud_ratio: float = 0.1,
    train_year_cutoff: int = 2018,
    validation_year: int = 2018,
    one_hot_threshold: int = 8,
) -> dict:
    """Run full preprocessing pipeline on PVC-mounted data.

    Reads raw data from PVC, processes it, and writes outputs to PVC.
    Combines: clean, encode, split, fit transformers, prepare GNN/XGB.

    Output structure on PVC:
        /data/
        ├── raw/card_transaction.parquet
        ├── processed/
        │   ├── train.parquet
        │   ├── validation.parquet
        │   └── test.parquet
        ├── transformers/
        │   ├── id_transformer.pkl
        │   └── feature_transformer.pkl
        └── gnn/
            ├── train_gnn/
            │   ├── edges/user_to_merchant.csv
            │   ├── nodes/{user,merchant}.csv
            │   └── ...
            └── test_gnn/
                └── ...

    Args:
        data_mount_path: Where data PVC is mounted
        under_sample: Whether to undersample majority class
        fraud_ratio: Target fraud ratio when undersampling
        train_year_cutoff: Year cutoff for training data
        validation_year: Year for validation data
        one_hot_threshold: Max categories for one-hot encoding

    Returns:
        Dict with preprocessing statistics
    """
    import os
    import pickle
    import numpy as np
    import pandas as pd
    from category_encoders import BinaryEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Constants
    COL_USER = "User"
    COL_CARD = "Card"
    COL_MERCHANT = "Merchant"
    COL_MCC = "MCC"
    COL_FRAUD = "Fraud"
    COL_YEAR = "Year"
    COL_AMOUNT = "Amount"
    COL_TIME = "Time"

    # Paths
    raw_path = os.path.join(data_mount_path, "raw", "card_transaction.parquet")
    processed_dir = os.path.join(data_mount_path, "processed")
    transformers_dir = os.path.join(data_mount_path, "transformers")
    gnn_dir = os.path.join(data_mount_path, "gnn")

    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(transformers_dir, exist_ok=True)
    os.makedirs(gnn_dir, exist_ok=True)

    # Step 1: Load and clean data
    print("Loading raw data...")
    df = pd.read_parquet(raw_path)
    original_count = len(df)

    # Rename columns
    df.columns = [c.split()[0] if " " in c else c for c in df.columns]

    # Clean amount
    if df[COL_AMOUNT].dtype == object:
        df[COL_AMOUNT] = df[COL_AMOUNT].str.replace("$", "").astype(float)

    # Encode fraud label
    df[COL_FRAUD] = df[COL_FRAUD].map({"Yes": 1, "No": 0}).fillna(0).astype(int)

    # Parse time to minutes
    def time_to_minutes(t):
        try:
            parts = str(t).split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except:
            return 0

    df[COL_TIME] = df[COL_TIME].apply(time_to_minutes)

    # Create card ID
    max_cards = df[COL_CARD].max() + 1
    df[COL_CARD] = df[COL_USER] * max_cards + df[COL_CARD]

    # Fill missing values
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("XX")
    df["Zip"] = df["Zip"].fillna(0)

    # Remove duplicates from non-fraud
    df_fraud = df[df[COL_FRAUD] == 1]
    df_non_fraud = df[df[COL_FRAUD] == 0].drop_duplicates(
        subset=[COL_CARD, COL_MERCHANT, COL_AMOUNT, COL_TIME, COL_YEAR]
    )
    df = pd.concat([df_fraud, df_non_fraud], ignore_index=True)
    after_dedup_count = len(df)

    # Undersample
    if under_sample:
        fraud_count = df[COL_FRAUD].sum()
        target_non_fraud = int(fraud_count / fraud_ratio) - fraud_count
        df_non_fraud = df[df[COL_FRAUD] == 0].sample(
            n=min(target_non_fraud, len(df[df[COL_FRAUD] == 0])),
            random_state=42
        )
        df = pd.concat([df[df[COL_FRAUD] == 1], df_non_fraud], ignore_index=True)

    final_count = len(df)
    fraud_count = df[COL_FRAUD].sum()
    print(f"Records: {original_count} -> {after_dedup_count} -> {final_count}")
    print(f"Fraud rate: {fraud_count/final_count:.4f}")

    # Step 2: Encode identifiers
    MERCHANT_AND_USER_COLS = [COL_MERCHANT, COL_MCC, COL_CARD]

    id_transformer = ColumnTransformer(
        transformers=[
            ("merchant_bin", BinaryEncoder(), [COL_MERCHANT]),
            ("mcc_bin", BinaryEncoder(), [COL_MCC]),
            ("card_bin", BinaryEncoder(), [COL_CARD]),
        ],
        remainder="drop",
    )
    id_transformer.fit(df[MERCHANT_AND_USER_COLS])

    with open(os.path.join(transformers_dir, "id_transformer.pkl"), "wb") as f:
        pickle.dump({"transformer": id_transformer, "columns": MERCHANT_AND_USER_COLS}, f)

    # Step 3: Split by year
    train_df = df[df[COL_YEAR] < train_year_cutoff].copy()
    val_df = df[df[COL_YEAR] == validation_year].copy()
    test_df = df[df[COL_YEAR] > validation_year].copy()

    train_df.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(processed_dir, "validation.parquet"), index=False)
    test_df.to_parquet(os.path.join(processed_dir, "test.parquet"), index=False)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Step 4: Fit feature transformers
    categorical_cols = ["Errors", "Use Chip"]
    numerical_cols = [COL_AMOUNT, COL_TIME]

    one_hot_cols = [c for c in categorical_cols if train_df[c].nunique() <= one_hot_threshold]
    binary_cols = [c for c in categorical_cols if train_df[c].nunique() > one_hot_threshold]

    transformers = []
    if one_hot_cols:
        transformers.append(("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), one_hot_cols))
    if binary_cols:
        transformers.append(("binary", BinaryEncoder(), binary_cols))
    if numerical_cols:
        transformers.append(("scaler", StandardScaler(), numerical_cols))

    feature_transformer = ColumnTransformer(transformers=transformers, remainder="drop")
    feature_transformer.fit(train_df)

    with open(os.path.join(transformers_dir, "feature_transformer.pkl"), "wb") as f:
        pickle.dump({"transformer": feature_transformer}, f)

    # Step 5: Prepare GNN data
    def prepare_gnn_split(data, split_name):
        split_dir = os.path.join(gnn_dir, f"{split_name}_gnn")
        edges_dir = os.path.join(split_dir, "edges")
        nodes_dir = os.path.join(split_dir, "nodes")
        os.makedirs(edges_dir, exist_ok=True)
        os.makedirs(nodes_dir, exist_ok=True)

        users = data[COL_CARD].unique()
        merchants = data[COL_MERCHANT].unique()
        user_to_id = {u: i for i, u in enumerate(sorted(users))}
        merchant_to_id = {m: i for i, m in enumerate(sorted(merchants))}

        edge_df = pd.DataFrame({
            "src": data[COL_CARD].map(user_to_id),
            "dst": data[COL_MERCHANT].map(merchant_to_id),
        })
        edge_df.to_csv(os.path.join(edges_dir, "user_to_merchant.csv"), index=False)

        label_df = pd.DataFrame({COL_FRAUD: data[COL_FRAUD].values})
        label_df.to_csv(os.path.join(edges_dir, "user_to_merchant_label.csv"), index=False)

        edge_features = feature_transformer.transform(data)
        pd.DataFrame(edge_features).to_csv(os.path.join(edges_dir, "user_to_merchant_attr.csv"), index=False)

        # Simplified node features
        user_data = data[[COL_CARD]].drop_duplicates().sort_values(by=COL_CARD)
        pd.DataFrame({"id": range(len(user_data))}).to_csv(os.path.join(nodes_dir, "user.csv"), index=False)

        merchant_data = data[[COL_MERCHANT]].drop_duplicates().sort_values(by=COL_MERCHANT)
        pd.DataFrame({"id": range(len(merchant_data))}).to_csv(os.path.join(nodes_dir, "merchant.csv"), index=False)

        print(f"  {split_name}: {len(edge_df)} edges, {len(users)} users, {len(merchants)} merchants")

    print("Preparing GNN data...")
    prepare_gnn_split(train_df, "train")
    prepare_gnn_split(test_df, "test")

    return {
        "original_records": original_count,
        "final_records": final_count,
        "fraud_rate": float(fraud_count / final_count),
        "train_records": len(train_df),
        "test_records": len(test_df),
    }


@dsl.component(base_image="python:3.12")
def prepare_training_config_on_pvc(
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
    """Write training config to PVC."""
    import json
    import os

    config = {
        "paths": {"data_dir": "/data/gnn", "output_dir": "/trained_models"},
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

    config_path = os.path.join(data_mount_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {config_path}")


@dsl.container_component
def run_nvidia_training_on_pvc():
    """Run NVIDIA training container with PVCs mounted."""
    return dsl.ContainerSpec(
        image="nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0",
        command=["/bin/bash", "-c"],
        args=[
            "ln -sf /data/config.json /app/config.json && "
            "cat /app/config.json && "
            "echo '=== GNN Data ===' && ls -la /data/gnn/ && "
            "cd /app && python main.py && "
            "echo '=== Training Complete ===' && ls -la /trained_models/"
        ],
    )


@dsl.component(base_image="python:3.12", packages_to_install=["boto3==1.34.0"])
def upload_model_to_s3(
    model_mount_path: str = "/trained_models",
    s3_prefix: str = "model-repository",
    s3_region: str = "us-east-1",
) -> dict:
    """Upload trained model from PVC to S3."""
    import os
    import boto3
    from pathlib import Path

    s3_bucket = os.environ.get("MODEL_BUCKET", "")
    if not s3_bucket:
        raise ValueError("MODEL_BUCKET environment variable not set")

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
            s3.upload_file(local_path, s3_bucket, s3_key)
            uploaded += 1
            total_bytes += size

    s3_uri = f"s3://{s3_bucket}/{s3_prefix}"
    print(f"\nUploaded {uploaded} files ({total_bytes} bytes) to {s3_uri}")
    return {"s3_uri": s3_uri, "files": uploaded, "bytes": total_bytes}


# =============================================================================
# End-to-End Pipeline
# =============================================================================


@dsl.pipeline(
    name="fraud-detection-e2e-pipeline",
    description="End-to-end fraud detection: S3 raw data -> PVC preprocessing -> "
    "PVC training -> S3 model output. Minimizes S3 usage with PVC for intermediates.",
)
def fraud_detection_e2e_pipeline(
    # S3 configuration
    s3_region: str = "us-east-1",
    source_path: str = "data/TabFormer/raw/card_transaction.v1.csv",
    s3_model_prefix: str = "model-repository",
    # Preprocessing parameters
    under_sample: bool = True,
    fraud_ratio: float = 0.1,
    train_year_cutoff: int = 2018,
    validation_year: int = 2018,
    one_hot_threshold: int = 8,
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
    # Validation
    run_smoke_test: bool = True,
    triton_service_name: str = "triton-inference-server",
    triton_namespace: str = "triton",
    triton_port: int = 8005,
):
    """End-to-end fraud detection pipeline.

    Default S3 source: s3://{S3_BUCKET}/data/TabFormer/raw/card_transaction.v1.csv

    Data flow:
    1. S3 -> Load raw TabFormer CSV to data PVC
    2. PVC -> Preprocess (clean, split, transform, prepare GNN)
    3. PVC -> Train GNN+XGBoost model (GPU)
    4. PVC -> Upload trained model to S3

    ConfigMap 'fraud-detection-config' must contain:
    - s3_bucket: S3 bucket for raw data and model output
    - model_bucket: S3 bucket for model output (can be same as s3_bucket)
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

    # Step 1: Load raw data from S3 to PVC
    load_task = load_raw_data_to_pvc(
        s3_region=s3_region,
        source_path=source_path,
        data_mount_path="/data",
    )
    load_task.after(data_pvc)
    kubernetes.mount_pvc(load_task, pvc_name=data_pvc.outputs["name"], mount_path="/data")
    kubernetes.use_config_map_as_env(
        load_task,
        config_map_name=CONFIG_MAP,
        config_map_key_to_env={"s3_bucket": "S3_BUCKET"},
    )

    # Step 2: Preprocess data on PVC
    preprocess_task = preprocess_data_on_pvc(
        data_mount_path="/data",
        under_sample=under_sample,
        fraud_ratio=fraud_ratio,
        train_year_cutoff=train_year_cutoff,
        validation_year=validation_year,
        one_hot_threshold=one_hot_threshold,
    )
    preprocess_task.after(load_task)
    kubernetes.mount_pvc(preprocess_task, pvc_name=data_pvc.outputs["name"], mount_path="/data")

    # Step 3: Write training config to PVC
    config_task = prepare_training_config_on_pvc(
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
    kubernetes.mount_pvc(config_task, pvc_name=data_pvc.outputs["name"], mount_path="/data")

    # Step 4: Train model (GPU)
    train_task = run_nvidia_training_on_pvc()
    train_task.after(config_task)
    train_task.after(model_pvc)
    kubernetes.mount_pvc(train_task, pvc_name=data_pvc.outputs["name"], mount_path="/data")
    kubernetes.mount_pvc(train_task, pvc_name=model_pvc.outputs["name"], mount_path="/trained_models")

    # GPU configuration
    kubernetes.add_node_selector(train_task, label_key="nvidia.com/gpu", label_value="true")
    kubernetes.add_toleration(train_task, key="nvidia.com/gpu", operator="Exists", effect="NoSchedule")
    train_task.set_accelerator_type("nvidia.com/gpu")
    train_task.set_accelerator_limit(1)

    # Step 5: Upload model to S3
    upload_task = upload_model_to_s3(
        model_mount_path="/trained_models",
        s3_prefix=s3_model_prefix,
        s3_region=s3_region,
    )
    upload_task.after(train_task)
    kubernetes.mount_pvc(upload_task, pvc_name=model_pvc.outputs["name"], mount_path="/trained_models")
    kubernetes.use_config_map_as_env(
        upload_task,
        config_map_name=CONFIG_MAP,
        config_map_key_to_env={"model_bucket": "MODEL_BUCKET"},
    )

    # Step 6: Cleanup PVCs
    kubernetes.DeletePVC(pvc_name=data_pvc.outputs["name"]).after(upload_task)
    kubernetes.DeletePVC(pvc_name=model_pvc.outputs["name"]).after(upload_task)

    # Step 7: Optional smoke test
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
        pipeline_func=fraud_detection_e2e_pipeline,
        package_path="fraud_detection_e2e_pipeline.yaml",
    )
    print("Pipeline compiled to fraud_detection_e2e_pipeline.yaml")
