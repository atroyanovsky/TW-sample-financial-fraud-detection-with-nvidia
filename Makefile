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

# Python tooling - prefer uv if available
UV := $(shell which uv 2>/dev/null)
ifdef UV
    PIP := uv pip
    PYTHON := uv run python
else
    PIP := pip
    PYTHON := python
endif

# Colors
YELLOW := \033[0;33m
GREEN := \033[0;32m
RED := \033[0;31m
NC := \033[0m

.PHONY: help check-deps deploy destroy kubeconfig dashboard notebook-apply \
        pipeline-compile pipeline-upload triton-status triton-models triton-build clean

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
	@echo "$(YELLOW)Configuration:$(NC)"
	@echo "  AWS_REGION=$(AWS_REGION)"
	@echo "  Python tooling: $(if $(UV),uv,pip)"
	@echo ""
	@echo "$(YELLOW)Example:$(NC)"
	@echo "  make deploy AWS_REGION=us-west-2"

#
# Dependency Checks
#
check-deps:
	@echo "$(YELLOW)Checking dependencies...$(NC)"
	@which aws > /dev/null || (echo "$(RED)✗ AWS CLI not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ AWS CLI$(NC)"
	@which kubectl > /dev/null || (echo "$(RED)✗ kubectl not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ kubectl$(NC)"
	@which node > /dev/null || (echo "$(RED)✗ Node.js not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ Node.js$(NC)"
	@which docker > /dev/null || (echo "$(RED)✗ Docker not found$(NC)" && exit 1)
	@echo "$(GREEN)✓ Docker$(NC)"
	@which uv > /dev/null && echo "$(GREEN)✓ uv$(NC)" || echo "$(YELLOW)○ uv not found (using pip)$(NC)"
	@aws sts get-caller-identity > /dev/null 2>&1 || (echo "$(RED)✗ AWS credentials not configured$(NC)" && exit 1)
	@echo "$(GREEN)✓ AWS credentials valid$(NC)"
	@echo ""
	@echo "$(GREEN)All required dependencies installed$(NC)"

#
# Infrastructure
#
deploy: check-deps
	@echo "$(YELLOW)Deploying CDK stacks...$(NC)"
	cd infra && npm install && npx cdk deploy --all --require-approval never
	@echo "$(GREEN)Deployment complete$(NC)"
	@$(MAKE) kubeconfig

destroy:
	@echo "$(RED)Destroying all infrastructure...$(NC)"
	cd infra && npx cdk destroy --all --force
	@echo "$(GREEN)Infrastructure destroyed$(NC)"

kubeconfig:
	@echo "$(YELLOW)Updating kubeconfig...$(NC)"
	aws eks update-kubeconfig --region $(AWS_REGION) --name $(CLUSTER_NAME)
	@echo "$(GREEN)kubeconfig updated$(NC)"

#
# Kubeflow
#
dashboard:
	@echo "$(YELLOW)Kubeflow Dashboard:$(NC)"
	@kubectl get svc -n deploykf-istio-gateway deploykf-gateway \
		-o jsonpath='{.status.loadBalancer.ingress[0].hostname}' 2>/dev/null \
		|| echo "Load balancer not ready"
	@echo ""
	@echo "Default credentials: user@example.com / user"

notebook-apply:
	@echo "$(YELLOW)Deploying Kubeflow notebook server...$(NC)"
	kubectl apply -f notebooks/kubeflow-notebook-server.yaml
	@echo "$(GREEN)Notebook server deployed to $(KF_NAMESPACE)$(NC)"

pipeline-compile:
	@echo "$(YELLOW)Compiling fraud detection pipeline...$(NC)"
	@echo "Using: $(if $(UV),uv,pip)"
	cd workflows && $(PIP) install -q kfp==2.10.1 kfp-kubernetes==1.4.0 && \
		$(PYTHON) -m workflows.cudf_e2e_pipeline
	@echo "$(GREEN)Pipeline compiled: workflows/fraud_detection_cudf_pipeline.yaml$(NC)"

pipeline-upload: pipeline-compile
	@echo "$(YELLOW)Upload the pipeline via Kubeflow UI:$(NC)"
	@echo "1. Open the dashboard: make dashboard"
	@echo "2. Navigate to Pipelines > Upload Pipeline"
	@echo "3. Upload: workflows/fraud_detection_cudf_pipeline.yaml"

#
# Triton
#
triton-status:
	@echo "$(YELLOW)Triton Inference Server Status:$(NC)"
	@kubectl get pods -n triton -l app.kubernetes.io/name=triton-inference-server
	@echo ""
	@kubectl get deployment -n triton

triton-models:
	@echo "$(YELLOW)Checking Triton models...$(NC)"
	@kubectl exec -n triton deploy/triton-server-triton-inference-server -- \
		curl -s localhost:8000/v2/models | python3 -m json.tool 2>/dev/null \
		|| echo "Triton not ready or no models loaded"

triton-build:
	@echo "$(YELLOW)Triggering Triton image rebuild...$(NC)"
	aws codebuild start-build --project-name triton-inference-image-build --region $(AWS_REGION)
	@echo "$(GREEN)Build started. Check CodeBuild console for progress.$(NC)"

#
# Development
#
clean:
	@echo "$(YELLOW)Cleaning up...$(NC)"
	-rm -f workflows/fraud_detection_cudf_pipeline.yaml
	-rm -rf infra/cdk.out
	-rm -rf infra/node_modules
	@echo "$(GREEN)Cleanup complete$(NC)"
