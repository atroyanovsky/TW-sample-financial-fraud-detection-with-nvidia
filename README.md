# NVIDIA Financial Fraud Detection Blueprint - AWS Edition

## Overview

This NVIDIA Financial Fraud Detection AI Blueprint provides a reference implementation for deploying an end-to-end fraud detection system using Graph Neural Networks (GNNs) on AWS. The solution leverages Kubeflow Pipelines on Amazon EKS for ML workflow orchestration and NVIDIA Triton Inference Server for high-performance model serving.

![Architecture diagram](./docs/arch-diagram.png)

### Architecture

1. **Kubeflow Pipelines** orchestrates the entire ML workflow on EKS, from data preprocessing to model deployment
2. **RAPIDS/cuDF** performs GPU-accelerated data preprocessing on the TabFormer dataset
3. **NVIDIA Training Container** trains a GNN+XGBoost ensemble model for fraud detection
4. **S3** stores raw data, and trained models
5. **NVIDIA Triton Inference Server** serves the trained model on GPU-enabled EKS nodes
6. **ArgoCD** manages GitOps-based deployments for infrastructure and model updates

## Prerequisites

1. **AWS Account** with permissions to create EKS, EC2, ECR, and S3 resources
2. **Local tools**: Docker, Node.js 20+, AWS CLI, kubectl

## Quick Start

### 1. Deploy Infrastructure

```bash
cd infra
npm install

# Bootstrap CDK (first time only)
npx cdk bootstrap aws://<ACCOUNT>/<REGION>

# Set environment
export CDK_DEFAULT_ACCOUNT=<your-account>
export CDK_DEFAULT_REGION=<your-region>

# Deploy all stacks
npx cdk deploy --all
```

This creates:
- EKS cluster with GPU node pools (Karpenter-managed)
- Kubeflow Pipelines for ML orchestration
- ArgoCD for GitOps deployments
- ECR repositories for training and inference images
- S3 buckets for data and models

### 2. Access Kubeflow Dashboard

```bash
# Update kubeconfig
aws eks update-kubeconfig --region <region> --name nvidia-fraud-detection-cluster-blueprint

# Get the External IP for the Kubeflow Proxy
kubectl get svc -n deploykf-istio-gateway deploykf-gateway
```

Then navigate to the external ip in your browser to get access to the Kubeflow UI.

### 3. Run the Pipeline

Use the Jupyter notebook in `notebooks/kubeflow-fraud-detection.ipynb` from a Kubeflow Notebook Server, or submit the compiled pipeline:

```bash
cd workflows
pip install kfp==2.10.1
python -m workflows.cudf_e2e_pipeline  # Compiles to YAML
# Upload fraud_detection_cudf_pipeline.yaml via Kubeflow UI
```

The pipeline:
1. Downloads TabFormer data from S3
2. Runs GPU-accelerated preprocessing with RAPIDS/cuDF
3. Trains GNN+XGBoost model
4. Uploads model to S3
5. Triton automatically loads the new model

## Project Structure

```
├── infra/                    # AWS CDK infrastructure
│   ├── lib/                  # CDK stack definitions
│   └── manifests/            # Helm charts and ArgoCD apps
├── notebooks/                # Kubeflow notebook for interactive development
├── workflows/                # Kubeflow Pipeline definitions
│   └── src/workflows/        # Pipeline components
├── triton/                   # Custom Triton Dockerfile
├── src/                      # Data preprocessing scripts
└── data/                     # Sample data and documentation
```

## Documentation

- [Kubeflow Complete Guide](./Kubeflow_Complete_Guide.md) - Deep dive into Kubeflow Pipelines
- [Infrastructure Overview](./infra/README.md) - CDK stack details
- [Pipeline Components](./docs/roadmap/03-pipeline-components.md) - Component documentation

## Cleanup

```bash
# Delete CloudFormation stacks
cd infra
npx cdk destroy --all

# Or delete specific stacks
aws cloudformation delete-stack --stack-name NvidiaFraudDetectionBlueprint
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.

## Authors

- Shardul Vaidya, AWS Partner Solutions Architect
- Zachary Jacobson, AWS Partner Solutions Architect
- Ragib Ahsan, AWS AI Acceleration Architect
