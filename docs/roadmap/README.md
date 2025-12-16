# Kubeflow Migration Roadmap

This roadmap guides the migration of the NVIDIA Financial Fraud Detection project from a SageMaker-based workflow to Kubeflow Pipelines on EKS. It is designed to be executed by AI agents working on discrete, well-defined tasks.

## Current Architecture

The project currently operates as follows: SageMaker notebooks handle development and preprocessing, SageMaker training jobs execute GNN model training using the NVIDIA financial-fraud-training container, trained models are stored in S3, and an EKS cluster running Triton Inference Server serves predictions. ArgoCD manages GitOps deployments, and CDK provisions all AWS infrastructure.

## Target Architecture

After migration: Kubeflow Pipelines orchestrates the entire ML workflow on the existing EKS cluster. Kubeflow notebooks replace SageMaker notebooks for development. Training runs on EKS GPU nodes (g4dn instances) managed by Karpenter. Triton deployments continue via ArgoCD with pipeline-triggered syncs. Katib provides hyperparameter optimization. All components share the same Kubernetes infrastructure.

## Roadmap Documents

| Document | Purpose | Estimated Effort |
|----------|---------|------------------|
| [01-project-overview.md](./01-project-overview.md) | Current state analysis, component inventory, data flows | Reference |
| [02-kubeflow-installation.md](./02-kubeflow-installation.md) | EKS cluster preparation and Kubeflow deployment | Week 1-2 |
| [03-pipeline-components.md](./03-pipeline-components.md) | KFP component definitions with code templates | Week 2-3 |
| [04-migration-phases.md](./04-migration-phases.md) | Phased migration plan with checkpoints | Week 1-8 |
| [05-infrastructure-changes.md](./05-infrastructure-changes.md) | CDK updates, IAM roles, Kubernetes manifests | Week 1-2 |
| [06-validation-testing.md](./06-validation-testing.md) | Testing strategy, success metrics, rollback plan | Ongoing |
| [07-kubeflow-notebooks.md](./07-kubeflow-notebooks.md) | Migrate SageMaker notebooks to Kubeflow-native notebooks | Week 9-10 |

## Quick Start for AI Agents

When picking up a task from this roadmap:

1. Read the relevant phase document completely before starting
2. Check the dependencies section to ensure prerequisite tasks are complete
3. Each task includes acceptance criteria - verify these before marking complete
4. Commit changes with conventional commit format: `feat(kubeflow): <description>`
5. Update task status in the phase document when complete

Key files to understand the current implementation:

- `README.md` - AWS deployment overview
- `Kubeflow_Complete_Guide.md` - Kubeflow reference (comprehensive)
- `notebooks/financial-fraud-usage.ipynb` - Current training workflow
- `workflows/workflows/pipeline.py` - KFP preprocessing pipeline
- `infra/lib/nvidia-fraud-detection-blueprint.ts` - EKS/CDK infrastructure

## Timeline Overview

```
Week 1-2: Infrastructure & Installation
├── Install Kubeflow on existing EKS cluster
├── Configure S3 artifact store
├── Set up IAM roles for pipeline execution
└── Validate base installation

Week 3-4: Core Pipeline Development
├── Create preprocessing component
├── Create training component (wrap NVIDIA container)
├── Create evaluation component
└── Test individual components

Week 5-6: Integration & Deployment
├── Wire components into full pipeline
├── Add quality gates and conditional logic
├── Configure ArgoCD sync trigger
└── Test end-to-end pipeline

Week 7-8: Optimization & Cutover
├── Add Katib for hyperparameter tuning
├── Set up scheduled pipeline runs
├── Performance validation against SageMaker baseline
└── Decommission SageMaker workflows

Week 9-10: Notebook Migration
├── Create Kubeflow-native notebooks
├── Implement KFP client integration
├── Port evaluation and inference testing
└── Documentation and user guides
```

## Success Criteria

The migration is complete when:

1. **Functional parity**: Pipeline produces models with equivalent accuracy to SageMaker workflow (AUC-ROC within 1%)
2. **Automation**: Full pipeline runs without manual intervention from data ingestion to model deployment
3. **Reproducibility**: Any pipeline run can be reproduced from its recorded parameters and artifacts
4. **Observability**: Metrics, logs, and artifacts are accessible via Kubeflow UI
5. **Cost efficiency**: Training costs are equal to or lower than SageMaker baseline

## Phase Dependencies

```
[Phase 1: Infrastructure] ──┬──> [Phase 2: Components]
                            │
[Phase 5: CDK Updates] ─────┘
                                      │
                                      v
                            [Phase 3: Pipeline Integration]
                                      │
                                      v
                            [Phase 4: Optimization & Katib]
                                      │
                                      v
                            [Phase 6: Validation & Cutover]
```

Phase 1 and Phase 5 can run in parallel. Phase 2 depends on Phase 1 completion. Phases 3-6 are sequential.

## Key Technical Decisions

These decisions have been made and should not be revisited without discussion:

- **KFP v2 SDK**: Use the v2 SDK with Python function-based components
- **S3 artifact store**: Reuse existing S3 bucket (`ml-on-containers-*`) for pipeline artifacts
- **Existing EKS cluster**: Deploy Kubeflow to the same cluster running Triton
- **Karpenter for GPU nodes**: Continue using Karpenter for g4dn instance scaling
- **ArgoCD for Triton**: Continue using ArgoCD for Triton deployments, triggered by pipeline

## Reference Materials

- `Kubeflow_Complete_Guide.md` in project root - comprehensive Kubeflow documentation
- [Kubeflow on AWS](https://awslabs.github.io/kubeflow-manifests/) - official AWS deployment guide
- [KFP SDK Reference](https://kubeflow-pipelines.readthedocs.io/) - API documentation
