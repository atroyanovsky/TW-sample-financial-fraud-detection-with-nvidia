# 04 - Migration Phases: SageMaker to Kubeflow

This document outlines the phased migration plan for transitioning the Financial Fraud Detection workflow from SageMaker to Kubeflow Pipelines. Each phase includes specific tasks, acceptance criteria, and rollback procedures.

## Phase Overview

| Phase | Timeline | Focus | Key Deliverable |
|-------|----------|-------|-----------------|
| 1 | Week 1-2 | Foundation | Kubeflow installed, GPU verified |
| 2 | Week 3-4 | Core Pipeline | Training runs on Kubeflow |
| 3 | Week 5-6 | End-to-End | Full pipeline with Triton deployment |
| 4 | Week 7-8 | Production Ready | Monitoring, scheduling, HPO |

## Phase 1: Foundation (Week 1-2)

**Objective:** Install Kubeflow on the existing EKS cluster and verify GPU workloads schedule correctly.

**Dependencies:** None (starting phase)

**Estimated Effort:** 16-24 hours

### Tasks

#### 1.1 Install Kubeflow on EKS (4-6 hours)

- [ ] Clone AWS Kubeflow manifests repository
- [ ] Configure S3 as artifact store
- [ ] Deploy Kubeflow using kustomize
- [ ] Verify all pods in `kubeflow` namespace are running

#### 1.2 Configure IAM for Kubeflow (2-3 hours)

- [ ] Create IAM role for pipeline runner
- [ ] Add S3 read/write permissions
- [ ] Create ServiceAccount with IRSA annotation
- [ ] Test S3 access from kubeflow namespace pod

#### 1.3 Verify GPU Scheduling (2-3 hours)

- [ ] Deploy test GPU pod with NVIDIA container
- [ ] Verify Karpenter provisions g4dn node
- [ ] Confirm `nvidia-smi` works inside pod

#### 1.4 Run Sample Pipeline (2-3 hours)

- [ ] Port-forward Kubeflow UI
- [ ] Upload and run "Hello World" pipeline
- [ ] Verify artifacts stored in S3

#### 1.5 Create Simple Custom Pipeline (4-6 hours)

- [ ] Write 2-component test pipeline
- [ ] Compile and upload via SDK
- [ ] Verify component outputs flow correctly

### Phase 1 Checkpoint

```bash
#!/bin/bash
echo "Checking Kubeflow pods..."
kubectl get pods -n kubeflow | grep -v Running && exit 1 || echo "All pods running"

echo "Checking GPU nodes..."
kubectl get nodes -l nvidia.com/gpu=true

echo "Checking S3 access..."
kubectl run s3-test --rm -i --restart=Never \
  --image=amazon/aws-cli \
  --serviceaccount=pipeline-runner \
  -n kubeflow \
  -- s3 ls s3://ml-on-containers/
```

### Phase 1 Rollback

1. Delete Kubeflow namespace: `kubectl delete namespace kubeflow`
2. Remove IAM roles created for Kubeflow
3. SageMaker workflow remains unaffected

## Phase 2: Core Pipeline (Week 3-4)

**Objective:** Implement preprocessing and training components that mirror the existing notebook workflow.

**Dependencies:** Phase 1 complete

**Estimated Effort:** 20-28 hours

### Tasks

#### 2.1 Containerize Preprocessing Code (4-6 hours)

- [ ] Extract logic from `src/preprocess_TabFormer.py`
- [ ] Create Dockerfile for preprocessing
- [ ] Build and push to ECR
- [ ] Test container standalone

#### 2.2 Create Preprocessing Component (3-4 hours)

- [ ] Wrap container as KFP component
- [ ] Define Input/Output artifacts
- [ ] Test component in isolation

#### 2.3 Create Training Component (4-6 hours)

- [ ] Use NVIDIA training container as base image
- [ ] Define model Output artifact
- [ ] Configure GPU resource requests
- [ ] Test with small dataset

#### 2.4 Build Core Pipeline (3-4 hours)

- [ ] Connect preprocessing to training
- [ ] Add pipeline parameters
- [ ] Compile pipeline
- [ ] Test data flow

#### 2.5 Validate Against SageMaker Baseline (4-6 hours)

- [ ] Run SageMaker training with fixed seed
- [ ] Run Kubeflow training with same seed
- [ ] Compare metrics (accuracy, F1, AUC)
- [ ] Document any differences

### Phase 2 Acceptance Criteria

| Metric | SageMaker | Kubeflow | Acceptable Variance |
|--------|-----------|----------|---------------------|
| Accuracy | X.XXX | X.XXX | +/-0.01 |
| F1 Score | X.XXX | X.XXX | +/-0.02 |
| AUC-ROC | X.XXX | X.XXX | +/-0.01 |
| Training Time | X min | X min | +/-20% |

### Phase 2 Rollback

1. Keep SageMaker training as primary
2. Delete failed pipeline runs
3. Container images remain in ECR for retry

## Phase 3: End-to-End Pipeline (Week 5-6)

**Objective:** Complete the pipeline with evaluation, S3 export, and Triton deployment integration.

**Dependencies:** Phase 2 complete

**Estimated Effort:** 24-32 hours

### Tasks

#### 3.1 Create Evaluation Component (4-5 hours)

- [ ] Implement metrics calculation
- [ ] Add quality gate logic
- [ ] Return pass/fail decision
- [ ] Log metrics to Kubeflow UI

#### 3.2 Create S3 Export Component (3-4 hours)

- [ ] Export model in Triton format
- [ ] Upload with versioned path
- [ ] Create Triton config.pbtxt
- [ ] Return model URI

#### 3.3 Create ArgoCD Trigger Component (4-5 hours)

- [ ] Implement ArgoCD sync via K8s API
- [ ] Wait for sync completion
- [ ] Handle sync failures gracefully

#### 3.4 Implement Conditional Logic (4-6 hours)

- [ ] Add quality gate condition
- [ ] Only deploy if evaluation passed
- [ ] Test both pass and fail scenarios

#### 3.5 Integration Testing (6-8 hours)

- [ ] Run full pipeline with real data
- [ ] Verify model deployed to Triton
- [ ] Send test inference request
- [ ] Validate inference results

### Phase 3 Checkpoint

- [ ] Full pipeline completes end-to-end
- [ ] Quality gate blocks bad models
- [ ] Good models deploy automatically
- [ ] Inference endpoint returns valid predictions

### Phase 3 Rollback

1. Revert ArgoCD to previous Triton deployment
2. Manual model deployment via existing process
3. Core pipeline (Phase 2) remains as fallback

## Phase 4: Production Ready (Week 7-8)

**Objective:** Add hyperparameter tuning, scheduling, monitoring, and documentation.

**Dependencies:** Phase 3 complete

**Estimated Effort:** 28-36 hours

### Tasks

#### 4.1 Set Up Katib for HPO (6-8 hours)

- [ ] Create Katib Experiment manifest
- [ ] Define search space (learning_rate, hidden_dim, dropout)
- [ ] Configure parallel trials
- [ ] Run HPO experiment

**Katib Configuration:**
```yaml
apiVersion: kubeflow.org/v1beta1
kind: Experiment
metadata:
  name: fraud-detection-hpo
spec:
  objective:
    type: maximize
    goal: 0.95
    objectiveMetricName: test_auc_roc
  algorithm:
    algorithmName: bayesian
  parallelTrialCount: 3
  maxTrialCount: 15
  parameters:
    - name: learning_rate
      parameterType: double
      feasibleSpace: {min: "0.0001", max: "0.01"}
    - name: hidden_dim
      parameterType: int
      feasibleSpace: {min: "64", max: "256", step: "32"}
```

#### 4.2 Configure Scheduled Runs (3-4 hours)

- [ ] Create recurring run configuration
- [ ] Set daily retraining schedule (2 AM UTC)
- [ ] Test schedule triggers correctly

```python
client.create_recurring_run(
    experiment_id=experiment.id,
    job_name='fraud-detection-daily',
    pipeline_package_path='pipeline.yaml',
    cron_expression='0 2 * * *',
    max_concurrency=1
)
```

#### 4.3 Implement Monitoring (6-8 hours)

- [ ] Export metrics to Prometheus
- [ ] Create Grafana dashboard
- [ ] Set up Slack alerts for failures
- [ ] Monitor GPU utilization

#### 4.4 Create Documentation (6-8 hours)

- [ ] Write operational runbook
- [ ] Document pipeline parameters
- [ ] Create troubleshooting guide
- [ ] Write onboarding guide

#### 4.5 Decommission SageMaker (4-6 hours)

- [ ] Verify Kubeflow handles all use cases
- [ ] Export historical metrics
- [ ] Update CI/CD to use Kubeflow
- [ ] Archive SageMaker notebooks

### Phase 4 Checkpoint

- [ ] HPO configured and tested
- [ ] Daily scheduled runs active
- [ ] Monitoring dashboard live
- [ ] Runbooks complete
- [ ] SageMaker deprecated

### Phase 4 Rollback

1. Re-enable SageMaker workflow
2. Pause Kubeflow scheduled runs
3. Route inference to previously deployed model

## Cross-Phase Dependencies

```
Phase 1 (Foundation)
    │
    v
Phase 2 (Core Pipeline)
    │
    +-- Depends on: Kubeflow installed, GPU scheduling working
    │
    v
Phase 3 (End-to-End)
    │
    +-- Depends on: Core pipeline tested, metrics validated
    │
    v
Phase 4 (Production)
    │
    +-- Depends on: Full pipeline deployed, Triton integration working
```

## Risk Mitigation

| Risk | Mitigation | Owner |
|------|------------|-------|
| GPU scheduling fails | Test Karpenter early; manual node scaling backup | Infra |
| Metrics mismatch | Parallel comparison; fixed seeds | ML |
| Triton deployment fails | Keep manual deployment; ArgoCD rollback | DevOps |
| Pipeline runs too long | Set timeouts; optimize containers | ML/Infra |

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Pipeline Success Rate | >95% | Kubeflow dashboard |
| Training Time | <=SageMaker | Pipeline metrics |
| Model Quality (AUC) | >0.90 | Evaluation component |
| Deployment Time | <10 min | End-to-end pipeline |
| Cost per Run | <=SageMaker | AWS Cost Explorer |

## AI Agent Execution Notes

When executing tasks:

1. **Task Isolation:** Each task can be executed independently within its phase
2. **Verification:** Run checkpoint scripts before proceeding to next phase
3. **Artifacts:** Commit all created files to the repository
4. **Documentation:** Update this document with actual values as tasks complete
5. **Blockers:** If blocked, document the issue and skip to parallel tasks

## Next Document

Proceed to [05-infrastructure-changes.md](./05-infrastructure-changes.md) for infrastructure updates.
