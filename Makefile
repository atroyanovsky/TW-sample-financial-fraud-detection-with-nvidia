# Makefile for GNN based financial fraud detection with Nvidia and AWS
# Handles infrastructure deployment, pipeline execution, and local development

# AWS Configuration
AWS_REGION ?= us-west-2
AWS_ACCOUNT_ID := $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
CLUSTER_NAME := nvidia-fraud-detection-cluster-blueprint

# Container Images
TRAINING_IMAGE := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/nvidia-training-repo:latest
TRITON_IMAGE := $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com/triton-inference-server:latest

# Kubeflow
KF_NAMESPACE := team-1

# Colors
YELLOW := \033[0;33m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m

.PHONY: help check-deps deploy destroy kubeconfig dashboard triton-status pipeline-run clean

help:
	@echo "$(YELLOW)NVIDIA Financial Fraud Detection - Kubeflow on EKS$(NC)"
	@echo ""
	@echo "$(GREEN)Infrastructure:$(NC)"
	@echo "  deploy             : Deploy all CDK stacks (EKS, Kubeflow, Triton)"
	@echo "  destroy            : Tear down all infrastructure"
	@echo "  kubeconfig         : Update kubeconfig for the cluster"
	@echo ""
	@echo "$(GREEN)Kubeflow:$(NC)"
	@echo "  dashboard          : Get Kubeflow dashboard URL"
	@echo "  notebook-apply     : Deploy the Kubeflow notebook server"
	@echo "  pipeline-compile   : Compile the fraud detection pipeline"
	@echo "  pipeline-upload    : Upload compiled pipeline to Kubeflow"
	@echo ""
	@echo "$(GREEN)Triton:$(NC)"
	@echo "  triton-status      : Check Triton Inference Server status"
	@echo "  triton-models      : List loaded models"
	@echo "  triton-build       : Trigger Triton image rebuild"
	@echo ""
	@echo "$(GREEN)Development:$(NC)"
	@echo "  check-deps         : Verify required tools are installed"
	@echo "  clean              : Clean up local artifacts"
	@echo ""
	@echo "$(YELLOW)Example:$(NC)"
	@echo "  make deploy AWS_REGION=us-west-2"

# Check dependencies
check-deps:
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	@which aws > /dev/null || (echo "$(RED)AWS CLI not found$(NC)" && exit 1)
	@which kubectl > /dev/null || (echo "$(RED)kubectl not found$(NC)" && exit 1)
	@which node > /dev/null || (echo "$(RED)Node.js not found$(NC)" && exit 1)
	@which docker > /dev/null || (echo "$(RED)Docker not found$(NC)" && exit 1)
	@aws sts get-caller-identity > /dev/null 2>&1 || (echo "$(RED)AWS credentials not configured$(NC)" && exit 1)
	@echo "$(GREEN)All dependencies installed$(NC)"

# Deploy infrastructure
deploy: check-deps
	@echo "$(YELLOW)Deploying CDK stacks...$(NC)"
	cd infra && npm install && npx cdk deploy --all --require-approval never
	@echo "$(GREEN)Deployment complete$(NC)"
	@$(MAKE) kubeconfig

# Destroy infrastructure
destroy:
	@echo "$(RED)Destroying all infrastructure...$(NC)"
	cd infra && npx cdk destroy --all --force
	@echo "$(GREEN)Infrastructure destroyed$(NC)"

# Update kubeconfig
kubeconfig:
	@echo "$(YELLOW)Updating kubeconfig...$(NC)"
	aws eks update-kubeconfig --region $(AWS_REGION) --name $(CLUSTER_NAME)
	@echo "$(GREEN)kubeconfig updated$(NC)"

# Get dashboard URL
dashboard:
	@echo "$(YELLOW)Kubeflow Dashboard:$(NC)"
	@kubectl get svc -n deploykf-istio-gateway deploykf-gateway -o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null || echo "Load balancer not ready"
	@echo ""
	@echo "Default credentials: user@example.com / user"

# Deploy notebook server
notebook-apply:
	@echo "$(YELLOW)Deploying Kubeflow notebook server...$(NC)"
	kubectl apply -f notebooks/kubeflow-notebook-server.yaml
	@echo "$(GREEN)Notebook server deployed to $(KF_NAMESPACE)$(NC)"

# Compile pipeline
pipeline-compile:
	@echo "$(YELLOW)Compiling fraud detection pipeline...$(NC)"
	cd workflows && pip install -q kfp==2.10.1 kfp-kubernetes==1.4.0 && python -m workflows.cudf_e2e_pipeline
	@echo "$(GREEN)Pipeline compiled: workflows/fraud_detection_cudf_pipeline.yaml$(NC)"

# Upload pipeline (requires port-forward or dashboard access)
pipeline-upload: pipeline-compile
	@echo "$(YELLOW)Upload the pipeline via Kubeflow UI:$(NC)"
	@echo "1. Open the dashboard: make dashboard"
	@echo "2. Navigate to Pipelines > Upload Pipeline"
	@echo "3. Upload: workflows/fraud_detection_cudf_pipeline.yaml"

# Triton status
triton-status:
	@echo "$(YELLOW)Triton Inference Server Status:$(NC)"
	@kubectl get pods -n triton -l app.kubernetes.io/name=triton-inference-server
	@echo ""
	@kubectl get deployment -n triton

# List Triton models
triton-models:
	@echo "$(YELLOW)Checking Triton models...$(NC)"
	@kubectl exec -n triton deploy/triton-server-triton-inference-server -- curl -s localhost:8000/v2/models | python3 -m json.tool 2>/dev/null || echo "Triton not ready or no models loaded"

# Trigger Triton image rebuild
triton-build:
	@echo "$(YELLOW)Triggering Triton image rebuild...$(NC)"
	aws codebuild start-build --project-name triton-inference-image-build --region $(AWS_REGION)
	@echo "$(GREEN)Build started. Check CodeBuild console for progress.$(NC)"

# Clean local artifacts
clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	-rm -f workflows/fraud_detection_cudf_pipeline.yaml
	-rm -rf infra/cdk.out
	-rm -rf infra/node_modules
	@echo "$(GREEN)Cleanup complete$(NC)"
