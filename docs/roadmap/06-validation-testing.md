# 06 - Validation and Testing Strategy

This document defines the testing and validation approach for the Kubeflow migration. Each test includes commands, expected outcomes, and pass/fail criteria.

## 1. Installation Validation

### 1.1 Kubeflow UI Accessible

```bash
kubectl port-forward -n kubeflow svc/ml-pipeline-ui 8080:80 &
curl -s -o /dev/null -w "%{http_code}" http://localhost:8080
```

**Pass:** HTTP 200, UI renders
**Fail:** Connection refused, timeout, or HTTP error

### 1.2 Pipeline Controller Running

```bash
kubectl get pods -n kubeflow -l app=ml-pipeline
kubectl logs -n kubeflow -l app=ml-pipeline --tail=50 | grep -i error
```

**Pass:** Pod Running 1/1, no errors
**Fail:** CrashLoopBackOff, ImagePullBackOff

### 1.3 Katib Controller Running

```bash
kubectl get pods -n kubeflow -l katib.kubeflow.org/component=controller
kubectl get crd | grep katib
```

**Pass:** Controller running, CRDs installed
**Fail:** Missing pods or CRDs

### 1.4 GPU Nodes Schedulable

```bash
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test
  namespace: default
spec:
  restartPolicy: Never
  containers:
  - name: cuda-test
    image: nvidia/cuda:12.0-base
    command: ["nvidia-smi"]
    resources:
      limits:
        nvidia.com/gpu: 1
  tolerations:
  - key: "nvidia.com/gpu"
    operator: "Exists"
    effect: "NoSchedule"
EOF

kubectl wait --for=condition=Ready pod/gpu-test --timeout=120s
kubectl logs gpu-test
kubectl delete pod gpu-test
```

**Pass:** Pod completes, nvidia-smi outputs GPU
**Fail:** Pod pending, no GPU allocated

## 2. Component Testing

### 2.1 S3 Access Test

```bash
kubectl run s3-test --rm -i --restart=Never \
  --image=amazon/aws-cli \
  --serviceaccount=kubeflow-pipeline-runner \
  -n kubeflow \
  -- s3 ls s3://ml-on-containers/
```

**Pass:** Bucket contents listed
**Fail:** Access denied

### 2.2 Component Unit Tests

```bash
pytest tests/components/test_preprocess.py -v
pytest tests/components/test_training.py -v
pytest tests/components/test_evaluate.py -v
```

**Pass:** All tests pass
**Fail:** Any test fails

### 2.3 Artifact Flow Test

Verify preprocessing output is valid training input:

```python
def test_artifact_flow():
    preprocess_output = pd.read_csv("train_dataset.csv")
    assert 'is_fraud' in preprocess_output.columns
    assert len(preprocess_output) > 0
```

## 3. Pipeline Validation

### 3.1 End-to-End Pipeline Run

```bash
kfp run submit \
  --experiment-name "validation" \
  --run-name "e2e-test" \
  --pipeline-package-path fraud_detection_pipeline.yaml \
  --params '{"s3_bucket": "ml-on-containers", "num_epochs": 10}'
```

**Pass:** All steps Succeeded
**Fail:** Any step Failed

### 3.2 Metrics Comparison

| Metric | SageMaker Baseline | Kubeflow | Acceptable Variance |
|--------|-------------------|----------|---------------------|
| AUC-ROC | 0.934 | X.XXX | +/- 0.01 |
| F1-Score | 0.866 | X.XXX | +/- 0.02 |
| Accuracy | 0.952 | X.XXX | +/- 0.01 |
| Training Time | 3600s | X s | +50% |

**Pass:** All within variance
**Fail:** Any metric outside threshold

### 3.3 Quality Gate Test

```python
def test_quality_gate_blocks_bad_model():
    # Run pipeline with low epochs (bad model)
    result = run_pipeline(num_epochs=1)
    assert result['eval_passed'] == False
    assert result['export_executed'] == False
    assert result['deploy_executed'] == False
```

## 4. Production Readiness

### 4.1 Concurrent Pipeline Runs

```bash
for i in {1..3}; do
  kfp run submit --run-name "load-test-$i" ... &
done
wait
```

**Pass:** All 3 runs complete
**Fail:** OOM, resource exhaustion

### 4.2 Failure Recovery

```bash
# Start pipeline
RUN_ID=$(kfp run submit ... -o json | jq -r '.run_id')

# Kill training pod mid-execution
sleep 120
TRAIN_POD=$(kubectl get pods -n kubeflow -l workflow=$RUN_ID -o name | head -1)
kubectl delete $TRAIN_POD -n kubeflow

# Verify pipeline fails gracefully
kfp run get $RUN_ID
```

**Pass:** Pipeline fails with clear error
**Fail:** Zombie workflow, unclear state

### 4.3 Cost Analysis

| Component | SageMaker | Kubeflow | Savings |
|-----------|-----------|----------|---------|
| Notebook Instance | $500/mo | $0 | $500 |
| Training Jobs | $36/mo | ~$50 | -$14 |
| EKS Cluster | $0 | $73/mo | -$73 |
| **Total** | $536/mo | $123/mo | **$413/mo** |

### 4.4 Security Scan

```bash
# Scan container images
trivy image nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0 --severity HIGH,CRITICAL

# Check RBAC
kubectl auth can-i --list --as=system:serviceaccount:kubeflow:kubeflow-pipeline-runner -n kubeflow

# Verify no privileged containers
kubectl get pods -n kubeflow -o json | jq '.items[].spec.containers[] | select(.securityContext.privileged == true)'
```

**Pass:** No critical CVEs, minimal RBAC, no privileged containers
**Fail:** Critical vulnerabilities, excessive permissions

## 5. Regression Testing

### 5.1 Daily Automated Pipeline

```yaml
apiVersion: argoproj.io/v1alpha1
kind: CronWorkflow
metadata:
  name: fraud-detection-daily
  namespace: kubeflow
spec:
  schedule: "0 2 * * *"
  timezone: "UTC"
  concurrencyPolicy: "Replace"
  workflowSpec:
    entrypoint: run-pipeline
    templates:
    - name: run-pipeline
      container:
        image: python:3.12
        command: ["python", "-c", "from kfp import Client; Client().create_run_from_pipeline_package(...)"]
```

### 5.2 Quality Monitoring

Track metrics over time and alert on degradation:

```python
THRESHOLDS = {
    'test_auc_roc': {'min': 0.90, 'warn': 0.92},
    'test_f1_score': {'min': 0.80, 'warn': 0.84}
}

def check_metrics(latest_metrics):
    for metric, thresholds in THRESHOLDS.items():
        value = latest_metrics.get(metric, 0)
        if value < thresholds['min']:
            send_alert(f"CRITICAL: {metric}={value}")
        elif value < thresholds['warn']:
            send_alert(f"WARNING: {metric}={value}")
```

## 6. Go/No-Go Checklist

### Phase 1: Installation

- [ ] Kubeflow UI accessible
- [ ] Pipeline controller running
- [ ] Katib controller running
- [ ] GPU nodes schedulable
- [ ] S3 access working

### Phase 2: Components

- [ ] Preprocessing component tests pass
- [ ] Training component tests pass
- [ ] Evaluation component tests pass
- [ ] GPU allocation verified

### Phase 3: Pipeline

- [ ] End-to-end pipeline succeeds
- [ ] AUC-ROC within 1% of baseline
- [ ] F1-Score within 2% of baseline
- [ ] Training time within 150% of baseline

### Phase 4: Production

- [ ] Concurrent runs succeed
- [ ] Failure recovery tested
- [ ] Cost analysis documented
- [ ] Security scan passed
- [ ] Daily runs scheduled
- [ ] Monitoring active

## Decision Criteria

### GO

All items checked, no critical blockers.

### NO-GO

Any of:
- AUC-ROC > 1% below baseline
- Critical security vulnerability
- Pipeline failure rate > 10%
- GPU scheduling consistently fails

### CONDITIONAL GO

May proceed with documented risk if:
- Minor metrics variance with remediation plan
- Non-critical security findings with 30-day timeline
- Performance within 200% with optimization plan

## Test Execution Report Template

```markdown
# Validation Report

**Date:** YYYY-MM-DD
**Executor:** [Name]
**Pipeline Version:** [Git SHA]

## Summary
- Total Tests: XX
- Passed: XX
- Failed: XX

## Metrics Comparison
| Metric | SageMaker | Kubeflow | Status |
|--------|-----------|----------|--------|
| AUC-ROC | 0.934 | X.XXX | PASS/FAIL |

## Blockers
1. [Description and remediation]

## Sign-off
- [ ] Engineering Lead
- [ ] ML Lead
- [ ] Security Review
```
