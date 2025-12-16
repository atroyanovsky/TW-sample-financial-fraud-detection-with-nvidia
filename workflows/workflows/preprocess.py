from kfp import dsl
from kfp.dsl import Dataset, Output, Input

@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=["boto3==1.34.0"]
)
def download_source_data(
        s3_bucket: str,
        s3_prefix: str,
        dataset_dir: Output[Dataset]
) -> dict:
    """
    Download TabFormer credit card transaction data from S3.

    s3_bucket: S3 bucket name
    s3_prefix: S3 prefix path to folder of data
    dataset_dir: Output directory -- managed by KFP
    """

    # creatively borrowed from --
    # https://stackoverflow.com/questions/72302266/is-it-possible-to-run-aws-s3-sync-with-boto3

    import os
    import boto3
    import datetime
    r"""
    Download the contents of a folder recursively into a directory

    Args:
        s3_path: the folder path in the s3 bucket
        local_dir: a relative or absolute directory path in the local file system
    """

    s3_path = s3_bucket + "/" + s3_prefix

    bucket_name, *path_parts = s3_path.split(os.sep)
    s3_folder = os.path.join(*path_parts)

    local_dir = dataset_dir.path

    s3_resource = boto3.resource('s3')
    s3_client = boto3.client('s3')
    bucket = s3_resource.Bucket(bucket_name)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))

        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue

        # getting metadata of s3 object
        meta_data = s3_client.head_object(Bucket=bucket.name, Key=obj.key)

        # checking whether s3 file is newer and need update
        if os.path.isfile(target):
            s3_last_modified = meta_data['LastModified'].replace(tzinfo=datetime.timezone.utc)
            local_last_modified = datetime.datetime.fromtimestamp(os.path.getmtime(target))
            local_last_modified = local_last_modified.replace(tzinfo=datetime.timezone.utc)
            if local_last_modified > s3_last_modified:
                continue

        tmp_target = f"{target}.tmp"
        bucket.download_file(obj.key, tmp_target)
        os.rename(tmp_target, target)

    return {
        "downloaded_from": f"s3://{bucket_name}/{s3_folder}",
        "size_downloaded": f"{os.path.getsize(local_dir) / (1024**2)} Mb",
        "kfp_artifact_uri": dataset_dir.uri
    }


@dsl.component(
    base_image="python:3.12-slim",
    packages_to_install=['pandas==2.0.0', 'numpy==1.24.0', 'scikit-learn==1.3.0', 'boto3==1.34.0',
                         "git+https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia.git@main#subdirectory=workflows"
                         ]
)
def preprocess_source(
        raw_dataset: Input[Dataset],
        user_mask_data: Output[Dataset],
        mx_mask_data: Output[Dataset],
        tx_mask_data: Output[Dataset],
) -> dict:
    """Preprocess source data for GNN training"""

    from workflows.workflows.utils.preprocess_TabFormer_lp import preprocess_data

    user_mask_map, mx_mask_map, tx_mask_map = preprocess_data(raw_dataset.path)







