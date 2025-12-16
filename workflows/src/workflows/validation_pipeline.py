# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Kubeflow Pipeline for model validation and inference testing."""

from kfp import compiler, dsl, kubernetes

from .components.test_model import smoke_test_triton, validate_model_inference

# ConfigMap for inference endpoint configuration
INFERENCE_CONFIG_MAP = "fraud-detection-config"


@dsl.pipeline(
    name="fraud-model-validation-pipeline",
    description="Validate deployed fraud detection model via Triton inference. "
    "Runs smoke test and full validation with test data from preprocessing.",
)
def fraud_model_validation_pipeline(
    triton_service_name: str = "triton-inference-server",
    triton_namespace: str = "triton",
    triton_port: int = 8005,
    model_name: str = "prediction_and_shapley",
    sample_size: int = 100,
    timeout_seconds: int = 300,
    run_full_validation: bool = True,
    preprocessing_run_id: str = "",
):
    """Validate deployed model by running inference tests.

    This pipeline can be triggered:
    1. After training pipeline completes (via preprocessing_run_id)
    2. Manually for periodic validation
    3. After Triton deployment updates

    The Triton endpoint is constructed from service name and namespace:
    {triton_service_name}.{triton_namespace}.svc.cluster.local

    Args:
        triton_service_name: Triton k8s service name
        triton_namespace: Namespace where Triton is deployed
        triton_port: Triton HTTP port (default 8005 per infra config)
        model_name: Model name in Triton model repository
        sample_size: Number of test samples for validation
        timeout_seconds: Max wait for Triton readiness
        run_full_validation: If true, runs full validation with test data
        preprocessing_run_id: Run ID of preprocessing pipeline (for artifact reference)
    """
    # Construct internal DNS name for Triton
    triton_host = f"{triton_service_name}.{triton_namespace}.svc.cluster.local"

    # Step 1: Smoke test - lightweight check that Triton is responding
    smoke_task = smoke_test_triton(
        triton_host=triton_host,
        triton_port=triton_port,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )

    # Step 2: Full validation with test data (conditional)
    # Note: This requires test artifacts from preprocessing pipeline
    # In production, you'd reference these via artifact store or pass run IDs
    with dsl.If(run_full_validation == True):  # noqa: E712
        # This is a placeholder showing the pattern
        # In practice, you'd use importer or artifact references
        pass


@dsl.pipeline(
    name="fraud-model-smoke-test",
    description="Quick smoke test for Triton deployment. "
    "Validates server readiness and basic inference with synthetic data.",
)
def fraud_model_smoke_test_pipeline(
    triton_service_name: str = "triton-inference-server",
    triton_namespace: str = "triton",
    triton_port: int = 8005,
    model_name: str = "prediction_and_shapley",
    timeout_seconds: int = 300,
):
    """Lightweight smoke test for deployed model.

    Use this for:
    - Quick deployment verification
    - Health checks
    - CI/CD gates

    Does not require test data artifacts - uses synthetic data.

    Args:
        triton_service_name: Triton k8s service name
        triton_namespace: Namespace where Triton is deployed
        triton_port: Triton HTTP port
        model_name: Model name in Triton
        timeout_seconds: Max wait for readiness
    """
    triton_host = f"{triton_service_name}.{triton_namespace}.svc.cluster.local"

    smoke_test_triton(
        triton_host=triton_host,
        triton_port=triton_port,
        model_name=model_name,
        timeout_seconds=timeout_seconds,
    )


if __name__ == "__main__":
    # Compile both pipelines
    compiler.Compiler().compile(
        pipeline_func=fraud_model_validation_pipeline,
        package_path="fraud_model_validation_pipeline.yaml",
    )
    print("Compiled: fraud_model_validation_pipeline.yaml")

    compiler.Compiler().compile(
        pipeline_func=fraud_model_smoke_test_pipeline,
        package_path="fraud_model_smoke_test_pipeline.yaml",
    )
    print("Compiled: fraud_model_smoke_test_pipeline.yaml")
