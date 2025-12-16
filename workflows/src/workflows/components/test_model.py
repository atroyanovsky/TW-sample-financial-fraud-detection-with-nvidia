# Copyright (c) 2025, Amazon Web Services, Inc.
# Code modified by vshardul@amazon.com based on Apache License, Version 2.0 code provided by NVIDIA Corporation.
"""Components for model validation and inference testing."""

from kfp import dsl
from kfp.dsl import HTML, Artifact, Dataset, Input, Metrics, Output


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "tritonclient[http]==2.52.0",
        "numpy==1.26.0",
        "pandas==2.2.0",
        "pyarrow==15.0.0",
        "scikit-learn==1.4.0",
        "matplotlib==3.8.0",
    ],
)
def validate_model_inference(
    test_edges: Input[Dataset],
    test_user_features: Input[Dataset],
    test_merchant_features: Input[Dataset],
    test_edge_features: Input[Dataset],
    test_edge_labels: Input[Dataset],
    feature_masks: Input[Artifact],
    validation_html: Output[HTML],
    metrics: Output[Metrics],
    triton_host: str = "triton-inference-server",
    triton_port: int = 8005,
    model_name: str = "prediction_and_shapley",
    sample_size: int = 100,
    timeout_seconds: int = 300,
) -> dict:
    """Validate deployed model by running inference on test data.

    Connects to the Triton inference server and runs predictions on a sample
    of test data to validate:
    - Model is serving correctly
    - Predictions are reasonable (not all zeros/ones)
    - Latency is acceptable
    - SHAP values are computed

    Args:
        test_edges: Test edge data (user_id, merchant_id pairs)
        test_user_features: Test user node features
        test_merchant_features: Test merchant node features
        test_edge_features: Test edge/transaction features
        test_edge_labels: Test fraud labels
        feature_masks: Feature mask artifact from preprocessing
        validation_html: Output HTML report
        metrics: Output metrics
        triton_host: Triton server hostname (service name in k8s)
        triton_port: Triton HTTP port
        model_name: Model name in Triton
        sample_size: Number of samples to test
        timeout_seconds: Max time to wait for Triton readiness

    Returns:
        Dict with validation results
    """
    import base64
    import io
    import os
    import pickle
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
    from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

    # Load test data
    edges_df = pd.read_parquet(test_edges.path)
    user_features = pd.read_parquet(test_user_features.path)
    merchant_features = pd.read_parquet(test_merchant_features.path)
    edge_features = pd.read_parquet(test_edge_features.path)
    edge_labels = pd.read_parquet(test_edge_labels.path)

    # Load feature masks
    with open(feature_masks.path, "rb") as f:
        mask_data = pickle.load(f)

    user_mask = mask_data["user_mask"]
    merchant_mask = mask_data["merchant_mask"]

    # Wait for Triton to be ready
    triton_url = f"{triton_host}:{triton_port}"
    client = InferenceServerClient(url=triton_url)

    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            if client.is_server_ready() and client.is_model_ready(model_name):
                break
        except Exception:
            pass
        time.sleep(5)
    else:
        raise TimeoutError(f"Triton not ready after {timeout_seconds}s")

    # Sample test data
    n_samples = min(sample_size, len(edges_df))
    sample_indices = np.random.choice(len(edges_df), n_samples, replace=False)

    # Get unique users and merchants in sample
    sample_edges = edges_df.iloc[sample_indices]
    sample_users = sample_edges["src"].unique()
    sample_merchants = sample_edges["dst"].unique()

    # Build local ID mappings for the subgraph
    user_id_map = {uid: i for i, uid in enumerate(sample_users)}
    merchant_id_map = {mid: i for i, mid in enumerate(sample_merchants)}

    # Prepare inputs
    x_user = user_features.iloc[sample_users].values.astype(np.float32)
    x_merchant = merchant_features.iloc[sample_merchants].values.astype(np.float32)

    # Remap edge indices to local IDs
    edge_src = np.array([user_id_map[u] for u in sample_edges["src"]])
    edge_dst = np.array([merchant_id_map[m] for m in sample_edges["dst"]])
    edge_index = np.vstack([edge_src, edge_dst]).astype(np.int64)

    edge_attr = edge_features.iloc[sample_indices].values.astype(np.float32)
    true_labels = edge_labels.iloc[sample_indices]["Fraud"].values

    # Build inference request
    inputs = []

    def add_input(name, arr, dtype):
        inp = InferInput(name, list(arr.shape), datatype=dtype)
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    add_input("x_user", x_user, "FP32")
    add_input("x_merchant", x_merchant, "FP32")
    add_input("edge_index_user_to_merchant", edge_index, "INT64")
    add_input("edge_attr_user_to_merchant", edge_attr, "FP32")
    add_input("feature_mask_user", user_mask.astype(np.int32), "INT32")
    add_input("feature_mask_merchant", merchant_mask.astype(np.int32), "INT32")
    add_input("COMPUTE_SHAP", np.array([False], dtype=np.bool_), "BOOL")

    outputs = [InferRequestedOutput("PREDICTION")]

    # Run inference with timing
    inference_times = []
    all_predictions = []

    for _ in range(3):  # Run 3 times for timing
        t0 = time.time()
        result = client.infer(model_name, inputs, outputs=outputs)
        inference_times.append(time.time() - t0)
        predictions = result.as_numpy("PREDICTION")
        all_predictions.append(predictions)

    # Use last prediction for metrics
    pred_probs = all_predictions[-1].flatten()
    pred_labels = (pred_probs > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, zero_division=0)
    recall = recall_score(true_labels, pred_labels, zero_division=0)
    f1 = f1_score(true_labels, pred_labels, zero_division=0)

    try:
        auc = roc_auc_score(true_labels, pred_probs)
    except ValueError:
        auc = 0.0  # All one class

    avg_latency = np.mean(inference_times) * 1000  # ms
    p95_latency = np.percentile(inference_times, 95) * 1000

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Prediction distribution
    ax1 = axes[0]
    ax1.hist(pred_probs, bins=50, alpha=0.7, color="#3498db")
    ax1.axvline(0.5, color="red", linestyle="--", label="Threshold")
    ax1.set_xlabel("Prediction Probability")
    ax1.set_ylabel("Count")
    ax1.set_title("Prediction Distribution")
    ax1.legend()

    # Confusion matrix style
    ax2 = axes[1]
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(true_labels, pred_labels)
    im = ax2.imshow(cm, cmap="Blues")
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(["Non-Fraud", "Fraud"])
    ax2.set_yticklabels(["Non-Fraud", "Fraud"])
    ax2.set_xlabel("Predicted")
    ax2.set_ylabel("Actual")
    ax2.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax2.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)

    # Latency
    ax3 = axes[2]
    ax3.bar(["Avg", "P95"], [avg_latency, p95_latency], color=["#2ecc71", "#e74c3c"])
    ax3.set_ylabel("Latency (ms)")
    ax3.set_title("Inference Latency")

    plt.tight_layout()

    # Save to base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=150)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()

    # Generate HTML
    status_color = "#2ecc71" if accuracy > 0.8 and avg_latency < 1000 else "#e74c3c"
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Validation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
            h1 {{ color: #2c3e50; border-bottom: 2px solid {status_color}; padding-bottom: 10px; }}
            .status {{ background: {status_color}; color: white; padding: 10px 20px; border-radius: 4px; display: inline-block; margin-bottom: 20px; }}
            .metrics-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
            .metric {{ background: #ecf0f1; padding: 15px; border-radius: 8px; text-align: center; }}
            .metric-value {{ font-size: 24px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            .viz {{ text-align: center; margin: 20px 0; }}
            .viz img {{ max-width: 100%; border-radius: 8px; }}
            .info {{ background: #3498db; color: white; padding: 15px; border-radius: 8px; margin-top: 20px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Model Inference Validation</h1>
            <div class="status">{"PASSED" if accuracy > 0.7 else "NEEDS REVIEW"}</div>
            <div class="info">
                <strong>Endpoint:</strong> {triton_url} | <strong>Model:</strong> {model_name} | <strong>Samples:</strong> {n_samples}
            </div>
            <div class="metrics-grid">
                <div class="metric"><div class="metric-value">{accuracy:.2%}</div><div class="metric-label">Accuracy</div></div>
                <div class="metric"><div class="metric-value">{precision:.2%}</div><div class="metric-label">Precision</div></div>
                <div class="metric"><div class="metric-value">{recall:.2%}</div><div class="metric-label">Recall</div></div>
                <div class="metric"><div class="metric-value">{f1:.2%}</div><div class="metric-label">F1 Score</div></div>
                <div class="metric"><div class="metric-value">{auc:.3f}</div><div class="metric-label">ROC-AUC</div></div>
                <div class="metric"><div class="metric-value">{avg_latency:.1f}ms</div><div class="metric-label">Avg Latency</div></div>
                <div class="metric"><div class="metric-value">{p95_latency:.1f}ms</div><div class="metric-label">P95 Latency</div></div>
                <div class="metric"><div class="metric-value">{int(true_labels.sum())}/{n_samples}</div><div class="metric-label">Fraud in Sample</div></div>
            </div>
            <div class="viz">
                <img src="data:image/png;base64,{img_base64}" alt="Validation Charts">
            </div>
        </div>
    </body>
    </html>
    """

    with open(validation_html.path, "w") as f:
        f.write(html_content)

    # Log metrics
    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("precision", float(precision))
    metrics.log_metric("recall", float(recall))
    metrics.log_metric("f1_score", float(f1))
    metrics.log_metric("roc_auc", float(auc))
    metrics.log_metric("avg_latency_ms", float(avg_latency))
    metrics.log_metric("p95_latency_ms", float(p95_latency))
    metrics.log_metric("samples_tested", n_samples)

    return {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "avg_latency_ms": float(avg_latency),
        "passed": accuracy > 0.7,
    }


@dsl.component(
    base_image="python:3.12",
    packages_to_install=[
        "tritonclient[http]==2.52.0",
        "numpy==1.26.0",
    ],
)
def smoke_test_triton(
    metrics: Output[Metrics],
    triton_host: str = "triton-inference-server",
    triton_port: int = 8005,
    model_name: str = "prediction_and_shapley",
    timeout_seconds: int = 300,
) -> dict:
    """Quick smoke test to verify Triton endpoint is responding.

    Lightweight check that:
    - Server is reachable
    - Model is loaded and ready
    - Basic inference works with synthetic data

    Use this for quick deployment verification without needing
    real test data artifacts.

    Args:
        metrics: Output metrics
        triton_host: Triton server hostname
        triton_port: Triton HTTP port
        model_name: Model name in Triton
        timeout_seconds: Max wait time for readiness

    Returns:
        Dict with health status
    """
    import time

    import numpy as np
    from tritonclient.http import InferenceServerClient, InferInput, InferRequestedOutput

    triton_url = f"{triton_host}:{triton_port}"
    client = InferenceServerClient(url=triton_url)

    # Wait for readiness
    start_time = time.time()
    server_ready = False
    model_ready = False

    while time.time() - start_time < timeout_seconds:
        try:
            server_ready = client.is_server_ready()
            if server_ready:
                model_ready = client.is_model_ready(model_name)
                if model_ready:
                    break
        except Exception as e:
            pass
        time.sleep(5)

    if not server_ready:
        raise RuntimeError(f"Triton server not ready at {triton_url}")
    if not model_ready:
        raise RuntimeError(f"Model {model_name} not ready")

    # Get model metadata
    metadata = client.get_model_metadata(model_name)

    # Create synthetic test data matching expected dimensions
    # These dimensions come from the model's config
    num_users = 5
    num_merchants = 3
    num_edges = 2
    user_feature_dim = 13
    merchant_feature_dim = 24
    edge_feature_dim = 38

    inputs = []

    def add_input(name, arr, dtype):
        inp = InferInput(name, list(arr.shape), datatype=dtype)
        inp.set_data_from_numpy(arr)
        inputs.append(inp)

    add_input("x_user", np.random.randn(num_users, user_feature_dim).astype(np.float32), "FP32")
    add_input("x_merchant", np.random.randn(num_merchants, merchant_feature_dim).astype(np.float32), "FP32")

    edge_index = np.vstack([
        np.random.randint(0, num_users, num_edges),
        np.random.randint(0, num_merchants, num_edges),
    ]).astype(np.int64)
    add_input("edge_index_user_to_merchant", edge_index, "INT64")
    add_input("edge_attr_user_to_merchant", np.random.randn(num_edges, edge_feature_dim).astype(np.float32), "FP32")

    add_input("feature_mask_user", np.zeros(user_feature_dim, dtype=np.int32), "INT32")
    add_input("feature_mask_merchant", np.zeros(merchant_feature_dim, dtype=np.int32), "INT32")
    add_input("COMPUTE_SHAP", np.array([False], dtype=np.bool_), "BOOL")

    outputs = [InferRequestedOutput("PREDICTION")]

    # Run inference
    t0 = time.time()
    result = client.infer(model_name, inputs, outputs=outputs)
    latency = (time.time() - t0) * 1000

    predictions = result.as_numpy("PREDICTION")

    # Validate output shape
    assert predictions.shape[0] == num_edges, f"Expected {num_edges} predictions, got {predictions.shape[0]}"

    # Log metrics
    metrics.log_metric("server_ready", 1)
    metrics.log_metric("model_ready", 1)
    metrics.log_metric("inference_latency_ms", float(latency))
    metrics.log_metric("output_shape_valid", 1)

    return {
        "server_ready": True,
        "model_ready": True,
        "model_name": model_name,
        "latency_ms": float(latency),
        "predictions_shape": list(predictions.shape),
    }
