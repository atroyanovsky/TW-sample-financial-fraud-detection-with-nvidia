# Copyright (c) 2025, NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0
"""Component: Load raw TabFormer data from S3 or local filesystem."""

from kfp import dsl
from kfp.dsl import Dataset, Output, Metrics


@dsl.component(
    base_image="python:3.12",
    packages_to_install=["pandas==2.2.0", "boto3==1.34.0", "pyarrow==15.0.0"],
)
def load_raw_data(
    source_path: str,
    raw_data: Output[Dataset],
    metrics: Output[Metrics],
    s3_bucket: str = "",
    s3_region: str = "us-east-1",
) -> dict:
    """Load raw TabFormer credit card transaction data.

    Supports both S3 and local filesystem sources. For S3, provide bucket name
    and use source_path as the key. For local, provide full path in source_path.

    Args:
        source_path: Path to CSV file (S3 key if s3_bucket provided, else local path)
        raw_data: Output artifact for raw dataset
        metrics: Output metrics artifact
        s3_bucket: Optional S3 bucket name (empty for local file)
        s3_region: AWS region for S3 access

    Returns:
        Dict with loading statistics
    """
    import pandas as pd
    import os

    if s3_bucket:
        import boto3
        s3 = boto3.client("s3", region_name=s3_region)
        local_tmp = "/tmp/raw_data.csv"
        print(f"Downloading s3://{s3_bucket}/{source_path}")
        s3.download_file(s3_bucket, source_path, local_tmp)
        data = pd.read_csv(local_tmp)
        os.remove(local_tmp)
    else:
        print(f"Loading local file: {source_path}")
        data = pd.read_csv(source_path)

    num_records = len(data)
    num_columns = len(data.columns)

    print(f"Loaded {num_records:,} records with {num_columns} columns")
    print(f"Columns: {list(data.columns)}")

    # Save as parquet for efficient downstream processing
    data.to_parquet(raw_data.path, index=False)

    # Log metrics
    metrics.log_metric("num_records", num_records)
    metrics.log_metric("num_columns", num_columns)

    return {
        "num_records": num_records,
        "num_columns": num_columns,
        "columns": list(data.columns),
        "source": f"s3://{s3_bucket}/{source_path}" if s3_bucket else source_path,
    }
