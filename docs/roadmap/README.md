# Kubeflow Migration Roadmap

This roadmap guides the migration of the NVIDIA Financial Fraud Detection project from a SageMaker-based workflow to Kubeflow Pipelines on EKS. It is designed to be executed by AI agents working on discrete, well-defined tasks.

## Current Status

**Migration Status: COMPLETE**

The migration from SageMaker to Kubeflow on EKS has been completed. All core infrastructure is deployed and operational.

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 1: Foundation | COMPLETE | deployKF installed, GPU nodes working |
| Phase 2: Core Pipeline | COMPLETE | cuDF preprocessing pipeline in `workflows/` |
| Phase 3: End-to-End | COMPLETE | Full pipeline with Triton deployment |
| Phase 4: Production Ready | COMPLETE | Notebooks migrated, Triton serving |
| Phase 5: Infrastructure | COMPLETE | CDK stacks deployed, ArgoCD configured |

## Current Architecture

The project now operates as follows: Kubeflow Pipelines orchestrates the ML workflow on EKS. Kubeflow notebooks provide development environments. Training runs on EKS GPU nodes (g4dn/g5 instances) managed by Karpenter. Custom Triton image serves the GNN+XGBoost model with Shapley explainability. ArgoCD manages GitOps deployments. CDK provisions all AWS infrastructure.

## Roadmap Documents

| Document | Purpose | Status |
|----------|---------|--------|
| [01-project-overview.md](./01-project-overview.md) | Current state analysis, component inventory | Reference |
| [02-kubeflow-installation.md](./02-kubeflow-installation.md) | EKS cluster preparation and Kubeflow deployment | COMPLETE |
| [03-pipeline-components.md](./03-pipeline-components.md) | KFP component definitions with code templates | COMPLETE |
| [04-migration-phases.md](./04-migration-phases.md) | Phased migration plan with checkpoints | COMPLETE |
| [05-infrastructure-changes.md](./05-infrastructure-changes.md) | CDK updates, IAM roles, Kubernetes manifests | COMPLETE |

## What Was Delivered

### Infrastructure
- deployKF (Kubeflow distribution) on EKS
- Custom Triton inference image with PyTorch, PyG, XGBoost, Captum
- ECR repository with CodeBuild auto-build on CDK deploy
- ArgoCD GitOps for Triton deployments
- Karpenter GPU node pools (g4dn, g5)
- NVIDIA GPU Operator v25.3.2

### Pipeline
- cuDF-accelerated preprocessing pipeline (`workflows/`)
- KFP v2 components for download, preprocess, train, upload
- PVC-based artifact passing between pipeline steps
- S3 integration for raw data and model artifacts

### Notebooks
- `notebooks/kubeflow-fraud-detection.ipynb` - Complete interactive pipeline
- Runs on Kubeflow Notebook Servers in `team-1` namespace
- Inline pipeline definition with KFP SDK
- Triton inference testing built-in

### Model Serving
- Triton Inference Server with custom image
- `prediction_and_shapley` model loaded and ready
- HTTP/gRPC endpoints on ports 8005/8006
- Shapley values for explainability

## Key Files

```
financial-fraud-detection/
├── README.md                           # Project overview
├── notebooks/
│   ├── kubeflow-fraud-detection.ipynb  # Main interactive notebook
│   └── kubeflow-notebook-server.yaml   # Notebook server manifest
├── workflows/
│   └── src/workflows/
│       ├── cudf_e2e_pipeline.py        # Pipeline definition
│       └── components/
│           └── preprocess_tabformer.py # Preprocessing component
├── infra/
│   ├── lib/
│   │   ├── nvidia-fraud-detection-blueprint.ts
│   │   └── triton-image-repo.ts        # ECR + CodeBuild
│   └── manifests/
│       ├── argocd/                     # ArgoCD applications
│       ├── helm/triton/                # Triton Helm chart
│       └── nginx-proxy.yaml            # Port-forward proxy
└── triton/
    └── Dockerfile                      # Custom Triton image
```

## Access Points

| Service | Access Method |
|---------|--------------|
| Kubeflow Dashboard | `kubectl port-forward -n deploykf-istio-gateway pod/nginx-proxy 8443:8443` then https://deploykf.example.com:8443 |
| Triton Server | `kubectl port-forward -n triton svc/triton-server-triton-inference-server 8005:8005` |
| ArgoCD | Via deploykf dashboard or direct port-forward |

## Key Technical Decisions

These decisions were made during migration:

- **deployKF**: Used instead of raw Kubeflow manifests for easier installation
- **KFP v2 SDK**: Python function-based components with `kfp-kubernetes` for PVC support
- **Custom Triton Image**: Required for PyTorch + torch_geometric + XGBoost + Captum
- **S3 Artifact Store**: Reused existing `ml-on-containers-*` buckets
- **ArgoCD for Triton**: Continues managing Triton via Helm charts
- **No SageMaker**: Fully removed - no hybrid approach
