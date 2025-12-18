import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from .sagemaker_inference import prepare_and_send_inference_request

def scan_thresholds(y, predictions, step=0.01):
    """
    Scan all thresholds from 0 to 1 and compute precision, recall, and F1 for each.
    Plots the curves and highlights the optimal F1 threshold and default 0.5 threshold.
    
    Args:
        y: Ground truth binary labels
        predictions: Prediction probabilities (0-1)
        step: Threshold step size (default 0.01)
    
    Returns:
        dict with thresholds, precision, recall, f1 arrays and best threshold info
    """
    thresholds = np.arange(0, 1 + step, step)
    precisions = []
    recalls = []
    f1_scores = []
    
    for thresh in thresholds:
        y_pred = (predictions > thresh).astype(int)
        precisions.append(precision_score(y, y_pred, zero_division=0))
        recalls.append(recall_score(y, y_pred, zero_division=0))
        f1_scores.append(f1_score(y, y_pred, zero_division=0))
    
    precisions = np.array(precisions)
    recalls = np.array(recalls)
    f1_scores = np.array(f1_scores)
    
    # Find best F1 threshold
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    # Get metrics at default 0.5 threshold
    default_idx = int(0.5 / step)
    default_f1 = f1_scores[default_idx]
    default_precision = precisions[default_idx]
    default_recall = recalls[default_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, precisions, label='Precision', color='blue', linewidth=2)
    ax.plot(thresholds, recalls, label='Recall', color='green', linewidth=2)
    ax.plot(thresholds, f1_scores, label='F1 Score', color='red', linewidth=2)
    
    # Mark best F1 threshold
    ax.axvline(x=best_threshold, color='red', linestyle='--', alpha=0.7, 
               label=f'Best F1 threshold: {best_threshold:.2f} (F1={best_f1:.4f})')
    ax.scatter([best_threshold], [best_f1], color='red', s=100, zorder=5, marker='*')
    
    # Mark default 0.5 threshold
    ax.axvline(x=0.5, color='gray', linestyle=':', alpha=0.7,
               label=f'Default threshold: 0.50 (F1={default_f1:.4f})')
    ax.scatter([0.5], [default_f1], color='gray', s=100, zorder=5, marker='o')
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Precision, Recall, and F1 Score vs Decision Threshold', fontsize=14)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\n" + "="*60)
    print("  THRESHOLD ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n  Best F1 Threshold: {best_threshold:.2f}")
    print(f"    - Precision: {precisions[best_idx]:.4f}")
    print(f"    - Recall:    {recalls[best_idx]:.4f}")
    print(f"    - F1 Score:  {best_f1:.4f}")
    print(f"\n  Default Threshold (0.50):")
    print(f"    - Precision: {default_precision:.4f}")
    print(f"    - Recall:    {default_recall:.4f}")
    print(f"    - F1 Score:  {default_f1:.4f}")
    print(f"\n  F1 Improvement: {(best_f1 - default_f1):.4f} ({(best_f1 - default_f1) / default_f1 * 100:.2f}%)" if default_f1 > 0 else "")
    print("="*60)
    
    return {
        'thresholds': thresholds,
        'precisions': precisions,
        'recalls': recalls,
        'f1_scores': f1_scores,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_precision': precisions[best_idx],
        'best_recall': recalls[best_idx]
    }


def compute_score_for_batch(y, predictions, decision_threshold = 0.5):
    # Apply threshold
    y_pred = (predictions > decision_threshold).astype(int)

    # Compute evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    # Confusion matrix
    classes = ['Non-Fraud', 'Fraud']
    columns = pd.MultiIndex.from_product([["Predicted"], classes])
    index = pd.MultiIndex.from_product([["Actual"], classes])

    conf_mat = confusion_matrix(y, y_pred)
    cm_df = pd.DataFrame(conf_mat, index=index, columns=columns)
    print(cm_df)

    # Plot the confusion matrix directly
    disp = ConfusionMatrixDisplay.from_predictions(
        y, y_pred, display_labels=classes
    )
    disp.ax_.set_title('Confusion Matrix')
    plt.show()

    # Print summary
    print("----Summary---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")


def measure_latency_for_samples(samples, compute_shap=False, warmup=5, host="localhost", http_port=8000):
    """Measure latency for a list of subgraph samples."""
    latencies = []
    
    # Warmup with first sample
    warmup_data = samples[0] | {"COMPUTE_SHAP": np.array([compute_shap], dtype=np.bool_)}
    for _ in range(warmup):
        prepare_and_send_inference_request(warmup_data, host=host, http_port=http_port)
    
    # Measure each sample
    for i, sample in enumerate(samples):
        request_data = sample | {"COMPUTE_SHAP": np.array([compute_shap], dtype=np.bool_)}
        
        start = time.perf_counter()
        prepare_and_send_inference_request(request_data, host=host, http_port=http_port)
        end = time.perf_counter()
        
        latencies.append((end - start) * 1000)  # Convert to ms
        
        if (i + 1) % 100 == 0:
            print(f"  Completed {i + 1}/{len(samples)} requests...")
    
    return latencies


def print_latency_stats(latencies, label=""):
    """Print latency statistics."""
    if len(latencies) < 2:
        print(f"Not enough samples for statistics (got {len(latencies)})")
        return
        
    latencies_sorted = sorted(latencies)
    n = len(latencies)
    
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Requests:    {n}")
    print(f"  Mean:        {statistics.mean(latencies):>8.2f} ms")
    print(f"  Std Dev:     {statistics.stdev(latencies):>8.2f} ms")
    print(f"  Min:         {min(latencies):>8.2f} ms")
    print(f"  Max:         {max(latencies):>8.2f} ms")
    print(f"  P50:         {latencies_sorted[int(n * 0.50)]:>8.2f} ms")
    print(f"  P90:         {latencies_sorted[int(n * 0.90)]:>8.2f} ms")
    print(f"  P95:         {latencies_sorted[int(n * 0.95)]:>8.2f} ms")
    print(f"  P99:         {latencies_sorted[min(int(n * 0.99), n-1)]:>8.2f} ms")
    print(f"{'='*55}")


def measure_throughput(samples, num_workers=10, compute_shap=False, host="localhost", http_port=8000):
    """Measure throughput with concurrent requests using realistic single-tx samples."""
    latencies = []
    errors = 0
    
    # Cycle through samples for each request
    sample_cycle = samples * (len(samples) // num_workers + 1)  # Ensure enough samples
    
    def send_request(sample):
        try:
            request_data = sample | {"COMPUTE_SHAP": np.array([compute_shap], dtype=np.bool_)}
            start = time.perf_counter()
            prepare_and_send_inference_request(request_data, host=host, http_port=http_port)
            end = time.perf_counter()
            return (end - start) * 1000, None
        except Exception as e:
            return None, str(e)
    
    num_requests = len(samples)
    print(f"Running throughput test: {num_requests} requests, {num_workers} concurrent workers...")
    
    overall_start = time.perf_counter()
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(send_request, sample_cycle[i]) for i in range(num_requests)]
        
        for future in as_completed(futures):
            latency, error = future.result()
            if error:
                errors += 1
            else:
                latencies.append(latency)
    
    overall_end = time.perf_counter()
    total_time = overall_end - overall_start
    
    successful = len(latencies)
    throughput = successful / total_time
    
    print(f"\n{'='*55}")
    print(f"  Throughput Test Results ({num_workers} workers)")
    print(f"{'='*55}")
    print(f"  Total requests:     {num_requests}")
    print(f"  Successful:         {successful}")
    print(f"  Errors:             {errors}")
    print(f"  Total time:         {total_time:.2f} s")
    print(f"  Throughput:         {throughput:.2f} tx/s")
    if latencies:
        print(f"  Avg latency:        {statistics.mean(latencies):.2f} ms")
        print(f"  P95 latency:        {sorted(latencies)[int(len(latencies) * 0.95)]:.2f} ms")
    print(f"{'='*55}")
    
    return throughput, latencies