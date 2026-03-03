import argparse

from sagemaker.core.workflow.parameters import (
    ParameterFloat,
    ParameterInteger,
    ParameterString,
)
from sagemaker.mlops.workflow.model_step import ModelStep
from sagemaker.mlops.workflow.pipeline import Pipeline
from sagemaker.mlops.workflow.steps import CacheConfig, TrainingStep
from sagemaker.serve.model_builder import ModelBuilder
from sagemaker.train import ModelTrainer
from sagemaker.train.configs import Compute, InputData, OutputDataConfig

try:
    from .sagemaker_fraud_detection_pipeline import get_session
except ImportError:
    from sagemaker_fraud_detection_pipeline import get_session


def get_amlsim_pipeline(
    region,
    role_arn,
    default_bucket,
    model_package_group_name="fraud-detection-models-amlsim",
    pipeline_name="FraudDetectionPipelineAMLSim",
    base_job_prefix="fraud-detect-amlsim",
    profile_name=None,
):
    """Create SageMaker pipeline for AMLSim-based GNN+XGBoost training."""
    sagemaker_session, pipeline_session = get_session(
        region, default_bucket, profile_name
    )
    account_id = sagemaker_session.account_id()

    cache_true_config = CacheConfig(enable_caching=True, expire_after="1d")
    cache_false_config = CacheConfig(enable_caching=False, expire_after="1d")

    # Parameters
    training_instance_count = ParameterInteger(name="TrainingInstanceCount", default_value=1)
    training_instance_type = ParameterString(
        name="TrainingInstanceType", default_value="ml.g5.xlarge"
    )

    gnn_input_s3_uri = ParameterString(
        name="GnnInputS3Uri",
        default_value=f"s3://{default_bucket}/data/amlsim/raw/amlsim_gnn",
    )
    model_repo_output_uri = ParameterString(
        name="ModelRepoOutputS3Uri",
        default_value=f"s3://{default_bucket}/model-repository/amlsim",
    )

    # GNN Hyperparameters
    gnn_hidden_channels = ParameterInteger(name="GnnHiddenChannels", default_value=32)
    gnn_n_hops = ParameterInteger(name="GnnNHops", default_value=2)
    gnn_layer = ParameterString(name="GnnLayer", default_value="SAGEConv")
    gnn_dropout_prob = ParameterFloat(name="GnnDropoutProb", default_value=0.1)
    gnn_batch_size = ParameterInteger(name="GnnBatchSize", default_value=4096)
    gnn_fan_out = ParameterInteger(name="GnnFanOut", default_value=10)
    gnn_num_epochs = ParameterInteger(name="GnnNumEpochs", default_value=8)

    # XGBoost Hyperparameters
    xgb_max_depth = ParameterInteger(name="XgbMaxDepth", default_value=6)
    xgb_learning_rate = ParameterFloat(name="XgbLearningRate", default_value=0.2)
    xgb_num_parallel_tree = ParameterInteger(name="XgbNumParallelTree", default_value=3)
    xgb_num_boost_round = ParameterInteger(name="XgbNumBoostRound", default_value=512)
    xgb_gamma = ParameterFloat(name="XgbGamma", default_value=0.0)

    # Step 1: Training (GNN+XGBoost) directly from AMLSim S3 inputs
    training_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/nvidia-training-repo-sagemaker:latest"
    )

    model_trainer = ModelTrainer(
        training_image=training_image_uri,
        compute=Compute(
            instance_type="ml.g5.xlarge",
            instance_count=1,
        ),
        base_job_name=f"{base_job_prefix}-train",
        sagemaker_session=pipeline_session,
        role=role_arn,
        hyperparameters={
            "model_kind": "GNN_XGBoost_NP",
            "gnn_hidden_channels": 32,
            "gnn_n_hops": 2,
            "gnn_layer": "TransformerConv",
            "gnn_dropout_prob": 0.2,
            "gnn_batch_size": 4096,
            "gnn_fan_out": 8,
            "gnn_metric": "f1",
            "gnn_num_epochs": 10,
            "gnn_weight_decay": 0.00001,
            "xgb_max_depth": 3,
            "xgb_learning_rate": 0.1,
            "xgb_num_parallel_tree": 50,
            "xgb_num_boost_round": 100,
            "xgb_gamma": 0.0,
        },
        input_data_config=[
            InputData(
                channel_name="gnn",
                data_source=gnn_input_s3_uri,
                content_type="text/csv",
            ),
        ],
        output_data_config=OutputDataConfig(s3_output_path=model_repo_output_uri),
    )

    train_args = model_trainer.train()

    step_train = TrainingStep(
        name="TrainModelAMLSim",
        step_args=train_args,
        cache_config=cache_false_config,
    )

    # Step 2: Register model
    triton_image_uri = (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/triton-inference-server:latest"
    )

    model_builder = ModelBuilder(
        s3_model_data_url=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri=triton_image_uri,
        sagemaker_session=pipeline_session,
        role_arn=role_arn,
    )

    step_register = ModelStep(
        name="RegisterModelAMLSim",
        step_args=model_builder.register(
            model_package_group_name=model_package_group_name,
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=[
                "ml.g4dn.xlarge",
                "ml.g4dn.2xlarge",
                "ml.g4dn.4xlarge",
                "ml.g4dn.8xlarge",
                "ml.g4dn.16xlarge",
                "ml.g5.xlarge",
                "ml.g5.2xlarge",
                "ml.g5.4xlarge",
                "ml.g5.8xlarge",
                "ml.g5.16xlarge",
                "ml.g6.xlarge",
                "ml.g6.2xlarge",
                "ml.g6.4xlarge",
                "ml.g6.8xlarge",
                "ml.g6.16xlarge",
                "ml.g6e.xlarge",
                "ml.g6e.2xlarge",
                "ml.g6e.4xlarge",
                "ml.g6e.8xlarge",
                "ml.g6e.16xlarge",
                "ml.p3.2xlarge",
            ],
            approval_status="PendingManualApproval",
        ),
    )

    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            training_instance_count,
            training_instance_type,
            gnn_input_s3_uri,
            model_repo_output_uri,
            gnn_hidden_channels,
            gnn_n_hops,
            gnn_layer,
            gnn_dropout_prob,
            gnn_batch_size,
            gnn_fan_out,
            gnn_num_epochs,
            xgb_max_depth,
            xgb_learning_rate,
            xgb_num_parallel_tree,
            xgb_num_boost_round,
            xgb_gamma,
        ],
        steps=[step_train, step_register],
        sagemaker_session=pipeline_session,
    )

    return pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-arn", required=True, help="SageMaker Execution Role ARN")
    parser.add_argument("--default-bucket", required=True, help="Default S3 bucket")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--profile", default=None, help="AWS profile name")
    parser.add_argument(
        "--pipeline-name",
        default="FraudDetectionPipelineAMLSim",
        help="Pipeline name",
    )
    parser.add_argument(
        "--model-package-group",
        default="fraud-detection-models-amlsim",
        help="Model package group name",
    )
    args = parser.parse_args()

    pipeline = get_amlsim_pipeline(
        region=args.region,
        role_arn=args.role_arn,
        default_bucket=args.default_bucket,
        model_package_group_name=args.model_package_group,
        pipeline_name=args.pipeline_name,
        profile_name=args.profile,
    )
    pipeline.upsert(role_arn=args.role_arn)
    print(f"Pipeline {pipeline.name} created/updated.")
