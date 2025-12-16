# 07 - Kubeflow Notebooks Migration

This document outlines the plan for migrating the existing SageMaker-oriented notebooks to Kubeflow-compatible notebooks that leverage the new workflow pipeline components.

## Current State

Two notebooks exist under `notebooks/`:

| Notebook | Purpose | Lines | Key Sections |
|----------|---------|-------|--------------|
| `financial-fraud-usage.ipynb` | Original SageMaker workflow | ~80KB | Environment setup, data prep, SageMaker training, Triton inference |
| `financial-fraud-usage-v2.ipynb` | Updated version | ~52KB | Similar structure, adds Brev support, local container training |

### Current Notebook Structure (v2)

1. **Environment Setup** - SageMaker Studio / Brev specific conda activation
2. **AWS Environment Variables** - CloudFormation stack outputs
3. **Data Download** - Manual IBM Box download instructions
4. **Directory Structure Check** - Verify data layout
5. **Training Configuration** - Create `training_config.json`
6. **Container Training** - Run NVIDIA training container locally or via SageMaker
7. **Triton Client Setup** - Install tritonclient
8. **Inference Testing** - Query Triton server, evaluate performance
9. **Shapley Values** - Explainability analysis

### Problems with Current Approach

| Issue | Impact |
|-------|--------|
| SageMaker-specific setup | Not portable to Kubeflow notebook servers |
| Manual data download | No integration with pipeline artifacts |
| Hardcoded paths | Breaks when run in different environments |
| Monolithic preprocessing | Duplicates logic now in `workflows/` |
| No pipeline integration | Can't trigger or monitor KFP runs |
| Training config as JSON file | Should use pipeline parameters |

## Target State

New notebooks under `notebooks/kubeflow/` that:

1. Run on Kubeflow Notebook Servers (JupyterLab on K8s)
2. Trigger pipeline runs via KFP SDK
3. Consume artifacts from completed pipeline runs
4. Provide interactive exploration and debugging
5. Support both full pipeline runs and individual component testing

### Proposed Notebook Structure

```
notebooks/
├── kubeflow/
│   ├── 01-data-exploration.ipynb      # Explore raw data, validate quality
│   ├── 02-run-preprocessing.ipynb     # Trigger preprocessing pipeline
│   ├── 03-training-experiments.ipynb  # Launch training, compare runs
│   ├── 04-model-evaluation.ipynb      # Analyze model performance
│   ├── 05-inference-testing.ipynb     # Test Triton endpoints
│   └── utils/
│       └── kfp_helpers.py             # Shared KFP client utilities
├── financial-fraud-usage.ipynb        # (keep for reference)
└── financial-fraud-usage-v2.ipynb     # (keep for reference)
```

## Notebook Specifications

### 01-data-exploration.ipynb

**Purpose:** Interactive exploration of raw TabFormer data before pipeline runs.

**Sections:**
1. Load raw CSV from local or S3
2. Basic statistics (record counts, column types)
3. Fraud rate analysis by year/month
4. Feature distributions
5. Correlation analysis (reuse `workflows/workflows/utils/transforms.py`)
6. Data quality checks

**Dependencies:** pandas, matplotlib, seaborn

**Artifacts Used:** None (works on raw data)

**Artifacts Produced:** None (exploration only)

### 02-run-preprocessing.ipynb

**Purpose:** Trigger and monitor the preprocessing pipeline.

**Sections:**
1. Connect to Kubeflow Pipelines
2. Configure pipeline parameters
3. Submit pipeline run
4. Monitor run progress
5. Retrieve output artifacts (XGB datasets, GNN graphs)
6. Validate outputs

**Key Code Pattern:**
```python
from kfp import Client
from workflows.workflows.pipeline import tabformer_preprocessing_pipeline

client = Client()  # Connects to in-cluster KFP

run = client.create_run_from_pipeline_func(
    tabformer_preprocessing_pipeline,
    arguments={
        "source_path": "./data/raw/card_transaction.v1.csv",
        "under_sample": True,
        "fraud_ratio": 0.1,
    },
    experiment_name="preprocessing-experiments",
)

# Monitor
client.wait_for_run_completion(run.run_id, timeout=3600)
```

**Artifacts Used:** Raw CSV data

**Artifacts Produced:** References to pipeline output artifacts

### 03-training-experiments.ipynb

**Purpose:** Launch GNN training runs with different hyperparameters.

**Sections:**
1. Load preprocessed artifacts from previous pipeline
2. Configure training parameters
3. Submit training pipeline/component
4. Compare multiple experiment runs
5. Visualize training curves
6. Select best model

**Integration Points:**
- Consumes: `gnn_train_*` artifacts from preprocessing
- Produces: Trained model artifacts
- Optional: Trigger Katib HPO experiment

**Key Code Pattern:**
```python
# Compare experiments
experiments = client.list_runs(experiment_id=exp.id)
for run in experiments:
    metrics = client.get_run(run.id).run.metrics
    print(f"{run.name}: AUC={metrics['test_auc_roc']}")
```

### 04-model-evaluation.ipynb

**Purpose:** Deep dive into model performance analysis.

**Sections:**
1. Load trained model and test data artifacts
2. Generate predictions
3. Confusion matrix analysis
4. ROC/AUC curves
5. Precision-Recall analysis
6. Error analysis (false positives/negatives)
7. Feature importance / Shapley values

**Artifacts Used:**
- Trained model from training pipeline
- Test datasets from preprocessing pipeline

### 05-inference-testing.ipynb

**Purpose:** Test deployed Triton inference endpoints.

**Sections:**
1. Discover Triton endpoint (via K8s Service/Ingress)
2. Health check
3. Single prediction test
4. Batch inference
5. Latency benchmarking
6. Shapley value computation
7. Compare with training metrics

**Key Code Pattern:**
```python
import tritonclient.http as httpclient

# Get endpoint from K8s service
endpoint = get_triton_endpoint("fraud-detection-triton")
client = httpclient.InferenceServerClient(url=endpoint)

# Inference
result = client.infer("fraud_detection", inputs=[...])
```

## Implementation Tasks

### Phase 1: Setup (4-6 hours)

- [ ] Create `notebooks/kubeflow/` directory structure
- [ ] Create `utils/kfp_helpers.py` with common functions:
  - `get_kfp_client()` - connect to Kubeflow Pipelines
  - `get_artifact_path(run_id, artifact_name)` - retrieve artifact URIs
  - `download_artifact(uri, local_path)` - fetch from S3
  - `get_triton_endpoint(service_name)` - discover Triton service endpoint
- [ ] Create base notebook template with standard imports

### Phase 2: Exploration Notebook (3-4 hours)

- [ ] Create `01-data-exploration.ipynb`
- [ ] Port relevant visualization code from existing notebooks
- [ ] Add correlation analysis using `workflows/workflows/utils/transforms.py`
- [ ] Test with sample data

### Phase 3: Pipeline Integration Notebooks (6-8 hours)

- [ ] Create `02-run-preprocessing.ipynb`
- [ ] Create `03-training-experiments.ipynb`
- [ ] Implement artifact retrieval and caching
- [ ] Add progress monitoring widgets

### Phase 4: Evaluation & Inference Notebooks (6-8 hours)

- [ ] Create `04-model-evaluation.ipynb`
- [ ] Port Shapley value analysis from existing notebooks
- [ ] Create `05-inference-testing.ipynb`
- [ ] Implement Triton client integration

### Phase 5: Documentation & Testing (4-6 hours)

- [ ] Add markdown documentation to each notebook
- [ ] Create README for `notebooks/kubeflow/`
- [ ] Test full workflow on Kubeflow notebook server
- [ ] Update main project README

## Migration Mapping

| Old Notebook Section | New Location | Notes |
|---------------------|--------------|-------|
| Environment Setup | Removed | Handled by Kubeflow notebook server |
| AWS Environment Variables | `utils/kfp_helpers.py` | Auto-discovered from K8s |
| Data Download | `02-run-preprocessing.ipynb` | Pipeline handles S3 |
| Directory Structure Check | `01-data-exploration.ipynb` | Simplified |
| Training Configuration | `03-training-experiments.ipynb` | Pipeline parameters |
| Container Training | `03-training-experiments.ipynb` | KFP component |
| Triton Client Setup | `05-inference-testing.ipynb` | K8s service discovery |
| Inference Testing | `05-inference-testing.ipynb` | Same logic, cleaner |
| Shapley Values | `04-model-evaluation.ipynb` | Consolidated |

## Dependencies

**Python packages for notebooks:**
```
kfp>=2.5.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
tritonclient[http]>=2.40.0
ipywidgets>=8.0.0
```

**Kubeflow requirements:**
- Kubeflow Notebook Server with GPU access (for evaluation)
- KFP v2 installed on cluster
- Triton deployed via ArgoCD (existing setup)
- IRSA configured for S3 access

## Success Criteria

- [ ] All 5 notebooks run successfully on Kubeflow notebook server
- [ ] Pipeline can be triggered and monitored from notebook
- [ ] Artifacts are correctly retrieved and used
- [ ] Triton inference works from notebook
- [ ] No SageMaker-specific code remains
- [ ] Documentation complete for new users

## Estimated Total Effort

| Phase | Hours |
|-------|-------|
| Setup | 4-6 |
| Exploration | 3-4 |
| Pipeline Integration | 6-8 |
| Evaluation & Inference | 6-8 |
| Documentation & Testing | 4-6 |
| **Total** | **23-32 hours** |

## Next Document

This completes the roadmap series. Return to [README.md](./README.md) for the full migration overview.
