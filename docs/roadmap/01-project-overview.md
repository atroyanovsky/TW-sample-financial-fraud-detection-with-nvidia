# 01 - Project Overview: Current State Analysis

## Purpose

This document describes the current architecture and workflow of the NVIDIA Financial Fraud Detection Blueprint on AWS. AI agents working on the Kubeflow migration should read this first to understand what exists today and what needs to change.

## System Overview

The Financial Fraud Detection system uses Graph Neural Networks (GNNs) to detect fraudulent financial transactions with high accuracy and reduced false positives. The current implementation runs on AWS using SageMaker for development/training and EKS for inference.

### Technology Stack

| Layer | Current Technology |
|-------|-------------------|
| Development | SageMaker Studio (JupyterLab) |
| Compute Instance | ml.g4dn.4xlarge (GPU) |
| Data Storage | Amazon S3 (`ml-on-containers` bucket) |
| Dataset | IBM TabFormer synthetic transactions |
| Training Container | `nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0` |
| Training Orchestration | SageMaker Training Jobs |
| Inference Runtime | NVIDIA Triton Inference Server |
| Inference Platform | Amazon EKS (Kubernetes 1.32) |
| GPU Nodes | g4dn.xlarge / g4dn.2xlarge via Karpenter |
| Infrastructure as Code | AWS CDK with EKS Blueprints |
| GitOps | ArgoCD |
| GPU Management | NVIDIA GPU Operator v25.3.2 |

## Current Workflow (Step-by-Step)

The workflow is manual and notebook-driven. Each step requires human intervention.

### Step 1: Environment Setup

**Location:** SageMaker Studio
**Action:** User creates conda environment from `conda/notebook_env.yaml`

```bash
conda env create -f conda/notebook_env.yaml
conda activate fraud_blueprint_env
```

### Step 2: Data Acquisition

**Location:** `notebooks/extra/download-tabformer.ipynb` or manual download
**Action:** Download TabFormer dataset from IBM Box, place in `data/TabFormer/raw/`

**Expected file:** `card_transaction.v1.csv`

### Step 3: Data Preprocessing

**Location:** `notebooks/financial-fraud-usage.ipynb`
**Code:** `src/preprocess_TabFormer.py`

**Key Functions:**
- `cramers_v(x, y)` - Statistical correlation for categorical features (lines 112-121)
- `create_feature_mask(columns)` - Feature selection mask generation (lines 124-148)
- `preprocess_data(tabformer_base_path)` - Main preprocessing pipeline (lines 151-963)

**Outputs:**
- Processed training data uploaded to S3
- Processed test data uploaded to S3
- Feature configuration files

### Step 4: Model Training

**Location:** SageMaker Training Job (triggered from notebook)
**Container:** `nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0`

**Process:**
1. Notebook calls SageMaker API to create training job
2. SageMaker pulls training container from ECR
3. Training job reads data from S3
4. GNN model trains on GPU instance
5. Trained model artifacts saved to S3

**Model Output Path:** `s3://ml-on-containers-{account}/models/fraud-detection/`

### Step 5: Model Export

**Action:** Training container produces Triton-compatible model artifacts
**Outputs:**
- Model weights (PyTorch/TorchScript format)
- Triton model configuration (`config.pbtxt`)
- Shapley value configuration for explainability

### Step 6: Model Deployment

**Location:** EKS Cluster
**Mechanism:** ArgoCD syncs Helm charts from `infra/manifests/helm/triton/`

**Process:**
1. Model artifacts uploaded to S3 trigger deployment
2. ArgoCD detects changes and syncs
3. Triton pods restart and load new model from S3
4. Load balancer routes traffic to Triton endpoints

### Step 7: Inference

**Endpoint:** ALB-backed Kubernetes Service
**Protocol:** HTTP/gRPC to Triton Inference Server

**Response includes:**
- Fraud probability score
- Shapley values for explainability

## Pain Points (Why Kubeflow?)

### Manual Orchestration
- Each notebook cell must be run manually
- No automated end-to-end pipeline
- Human must monitor and intervene at each step
- Re-running experiments requires manual tracking

### Environment Inconsistency
- SageMaker notebook environment differs from training container
- Preprocessing runs locally, training runs in container
- No guarantee of reproducibility across runs

### Limited Experiment Tracking
- No built-in experiment versioning
- Manual tracking of hyperparameters
- Difficult to compare model performance across runs

### Resource Inefficiency
- SageMaker notebook instance runs 24/7 (cost: ~$500/month)
- GPU resources not shared across workflows
- No automatic scaling during pipeline execution

### Disconnected Training and Serving
- Manual handoff between training completion and deployment
- No quality gates before deployment
- No automated rollback on model degradation

### No Hyperparameter Tuning Infrastructure
- Manual parameter selection
- Sequential experimentation only
- No parallel experiment execution

## Components to Migrate

### Must Migrate (Core Workflow)

| Component | Current | Target |
|-----------|---------|--------|
| Development Environment | SageMaker Studio | Kubeflow Notebook Servers |
| Workflow Orchestration | Manual notebook execution | Kubeflow Pipelines (KFP v2) |
| Training Execution | SageMaker Training Jobs | KFP Components on EKS GPU nodes |
| Experiment Tracking | None | KFP Metadata + Metrics |
| Hyperparameter Tuning | Manual | Katib |
| Model Serving Management | Manual ArgoCD sync | ArgoCD-triggered Triton reload |

### Code to Containerize

| File | Purpose | Target Component |
|------|---------|------------------|
| `src/preprocess_TabFormer.py` | Data preprocessing | `preprocess-component` |
| Training logic (in notebook) | GNN model training | `train-component` |
| Evaluation logic (in notebook) | Model evaluation | `evaluate-component` |
| S3 export logic | Model export | `export-component` |

### New Components to Create

| Component | Purpose |
|-----------|---------|
| `download-data` | Fetch TabFormer from source |
| `preprocess-data` | Run preprocessing pipeline |
| `train-gnn` | Execute GNN training |
| `evaluate-model` | Calculate metrics, quality gates |
| `export-to-s3` | Package model for Triton |
| `deploy-to-triton` | Trigger ArgoCD sync for Triton |

## What Stays the Same

These components are retained and integrated with Kubeflow:

### Amazon EKS Cluster
- **Keep:** Kubernetes 1.32 cluster
- **Keep:** Karpenter for GPU node autoscaling
- **Keep:** g4dn node pool configuration
- **Change:** Add Kubeflow namespace and components

### NVIDIA Triton Inference Server
- **Keep:** Triton deployment on EKS
- **Keep:** Helm charts in `infra/manifests/helm/triton/`
- **Keep:** ArgoCD manages deployment, pipeline triggers sync

### Amazon S3
- **Keep:** `ml-on-containers` bucket for data and models
- **Keep:** Same bucket structure
- **Add:** Kubeflow artifact storage prefix

### ArgoCD
- **Keep:** GitOps deployment mechanism
- **Keep:** Current ArgoCD installation
- **Change:** Add Kubeflow application manifests

### NVIDIA GPU Operator
- **Keep:** GPU Operator v25.3.2
- **Keep:** GPU device plugin configuration
- **No changes required**

### AWS CDK Infrastructure
- **Keep:** EKS Blueprints pattern
- **Keep:** VPC, IAM roles, security groups
- **Add:** Kubeflow IAM roles and service accounts

### Training Container
- **Keep:** `nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0`
- **Keep:** Same training logic
- **Change:** Invoke from KFP component instead of SageMaker

## Key Files Reference

AI agents should be familiar with these files:

```
financial-fraud-detection/
├── README.md                           # AWS deployment guide
├── Nvidia_README.md                    # Original NVIDIA instructions
├── Kubeflow_Complete_Guide.md          # Kubeflow reference (READ THIS)
├── conda/
│   └── notebook_env.yaml               # Python dependencies
├── data/
│   └── TabFormer/raw/                  # Dataset location
├── notebooks/
│   ├── financial-fraud-usage.ipynb     # Main workflow (MIGRATE THIS)
│   └── requirements.txt                # Notebook dependencies
├── src/
│   └── preprocess_TabFormer.py         # Preprocessing code (CONTAINERIZE)
├── infra/
│   ├── lib/
│   │   └── nvidia-fraud-detection-blueprint.ts  # CDK stack (EXTEND)
│   └── manifests/
│       ├── argocd/                     # ArgoCD bootstrap
│       └── helm/triton/                # Triton Helm chart
└── docs/
    └── roadmap/                        # Migration roadmap (YOU ARE HERE)
```

## Success Criteria

The migration is complete when:

1. **Pipeline Automation:** Full workflow runs without manual intervention
2. **Reproducibility:** Any pipeline run can be exactly reproduced
3. **Experiment Tracking:** All runs logged with parameters and metrics
4. **Quality Gates:** Models only deploy if evaluation thresholds pass
5. **Cost Reduction:** No always-on SageMaker notebook costs
6. **Hyperparameter Tuning:** Katib can run parallel experiments
7. **Parity:** Model accuracy matches or exceeds SageMaker baseline

## Next Document

Proceed to [02-kubeflow-installation.md](./02-kubeflow-installation.md) for EKS cluster preparation and Kubeflow installation tasks.
