# Changelog

All notable changes to the NVIDIA Financial Fraud Detection Blueprint are documented here.

## [2.0.0] - January 2025 (Latest)

Complete migration from SageMaker to Kubeflow Pipelines on EKS.

### New Features

- **Kubeflow Pipelines Integration**: Full ML pipeline orchestration via KFP v2 SDK with GPU-accelerated preprocessing and training stages
- **Interactive Notebook**: `notebooks/kubeflow-fraud-detection.ipynb` defines and submits pipelines directly from Kubeflow notebook servers
- **cuDF Preprocessing**: RAPIDS-based GPU preprocessing replaces pandas, reducing preprocessing time from hours to minutes on 24M transactions
- **Custom Triton Image**: ECR-hosted Triton image with PyTorch, torch_geometric, XGBoost, and Captum for Shapley explainability
- **CodeBuild Auto-Build**: Triton image automatically rebuilds on CDK deploy via Lambda-triggered CodeBuild project
- **PVC Artifact Passing**: Pipeline stages share data through persistent volumes instead of S3 round-trips
- **deployKF**: Kubeflow installation via deployKF Helm charts managed by ArgoCD

### Infrastructure Changes

- Added `triton-image-repo.ts` CDK stack for ECR repository and CodeBuild
- Added Kubeflow IAM roles with IRSA for S3 and ECR access
- Added `team-1` namespace with notebook server and pipeline runner service accounts
- Added nginx proxy for stable local port-forwarding during development
- Updated Karpenter node pools for GPU workloads (g4dn, g5, g6e support)
- Migrated ArgoCD to manage both Kubeflow and Triton deployments

### Removed

- SageMaker Training Jobs - replaced by KFP components on EKS
- SageMaker Notebook Instances - replaced by Kubeflow Notebook Servers
- `sagemaker-training-role.ts` and `sagemaker-notebook-role.ts` CDK stacks
- `sagemaker_config.json` configuration file

### Documentation

- Added `docs/roadmap/` with migration planning documents
- Added `docs/kubeflow-pipeline-journey.md` detailing the implementation journey
- Rewrote README as guided walkthrough with architecture explanations
- Added dashboard screenshots in `docs/img/`

## [1.0.0] - December 2024

Initial release with SageMaker-based training workflow.

### Features

- EKS cluster via CDK with EKS Blueprints
- Karpenter v1 for GPU node autoscaling (g4dn instances)
- NVIDIA GPU Operator v25.3.2
- Triton Inference Server deployment via ArgoCD and Helm
- SageMaker Training Jobs for GNN+XGBoost model training
- SageMaker Notebook Instance for development
- S3-based model registry with Triton auto-reload
- ALB Controller for load balancer provisioning

### Infrastructure

- VPC with 3 AZs and NAT gateway
- EKS 1.32 cluster with managed node groups
- ArgoCD for GitOps deployments
- Secrets Store CSI Driver integration
- IAM roles for SageMaker training and notebook access
