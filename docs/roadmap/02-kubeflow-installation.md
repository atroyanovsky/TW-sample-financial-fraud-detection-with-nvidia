# 02 - Kubeflow Installation on EKS

**STATUS: COMPLETE**

This document provides step-by-step tasks for installing Kubeflow on the existing EKS cluster. Each task is atomic and includes verification steps.

> Note: We used deployKF instead of raw Kubeflow manifests. The tasks below reflect the original plan; actual implementation used `deploykf` Helm charts via ArgoCD.

## Prerequisites

The existing infrastructure provides:

- EKS cluster running Kubernetes 1.32 (via EKS Blueprints)
- Karpenter v1 for node autoscaling
- GPU nodes: g4dn.xlarge/g4dn.2xlarge with NVIDIA GPU Operator v25.3.2
- ArgoCD installed and configured
- S3 bucket `ml-on-containers` for model artifacts
- AWS Load Balancer Controller

## Task 1: Prerequisites Check [COMPLETE]

**Objective:** Verify all required CLI tools are installed and configured.

### Commands

```bash
kubectl version --client
# Expected: Client Version: v1.28+

helm version
# Expected: version.BuildInfo{Version:"v3.12+"}

aws --version
# Expected: aws-cli/2.x.x

kubectl get nodes
# Expected: List of nodes in Ready state

aws sts get-caller-identity
# Expected: Valid account ID and ARN
```

### Verification

- [x] kubectl version >= 1.28
- [x] helm version >= 3.12
- [x] aws-cli version >= 2.0
- [x] kubectl can list nodes
- [x] AWS credentials valid for target account

## Task 2: Clone Kubeflow Manifests Repository [SKIPPED - Used deployKF]

**Objective:** Get the AWS-specific Kubeflow manifests.

### Commands

```bash
git clone https://github.com/awslabs/kubeflow-manifests.git
cd kubeflow-manifests
git checkout v1.8.0-aws-b1.0.0

export CLUSTER_NAME=$(kubectl config current-context | cut -d'/' -f2)
export CLUSTER_REGION=${CDK_DEFAULT_REGION:-us-east-1}
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "Cluster: $CLUSTER_NAME"
echo "Region: $CLUSTER_REGION"
echo "Account: $AWS_ACCOUNT_ID"
```

### Verification

- [x] Repository cloned successfully (used deployKF instead)
- [x] Correct branch/tag checked out
- [x] Environment variables set correctly

## Task 3: Configure S3 as Artifact Store [COMPLETE]

**Objective:** Configure Kubeflow Pipelines to use S3 for storing pipeline artifacts.

### Create Artifact Configuration

Create file `deployments/overlay/s3-artifact-config.yaml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: workflow-controller-configmap
  namespace: kubeflow
data:
  artifactRepository: |
    archiveLogs: true
    s3:
      bucket: ml-on-containers
      keyPrefix: kubeflow-artifacts/
      region: ${CLUSTER_REGION}
      endpoint: s3.amazonaws.com
      insecure: false
      useSDKCreds: true
```

### Commands

```bash
kubectl create namespace kubeflow --dry-run=client -o yaml | kubectl apply -f -
aws s3 ls s3://ml-on-containers/ --region $CLUSTER_REGION
```

### Verification

- [x] Namespace `kubeflow` exists
- [x] S3 bucket is accessible
- [x] Configuration files created

## Task 4: Configure RDS for Metadata (Optional) [SKIPPED - Using In-Cluster]

**Objective:** Set up RDS MySQL for Kubeflow metadata storage.

### Option A: Production (RDS)

```bash
aws rds create-db-instance \
  --db-instance-identifier kubeflow-metadata \
  --db-instance-class db.t3.medium \
  --engine mysql \
  --engine-version 8.0 \
  --master-username admin \
  --master-user-password <SECURE_PASSWORD> \
  --allocated-storage 20 \
  --no-publicly-accessible \
  --storage-encrypted
```

Create Kubernetes secret:

```bash
kubectl create secret generic rds-secret \
  --namespace kubeflow \
  --from-literal=username=admin \
  --from-literal=password=<SECURE_PASSWORD> \
  --from-literal=host=<RDS_ENDPOINT> \
  --from-literal=port=3306 \
  --from-literal=database=kubeflow
```

### Option B: Development (In-Cluster MySQL)

Skip RDS setup; Kubeflow deploys MySQL in-cluster by default.

### Verification

- [x] RDS instance available (Option A) OR skipped (Option B) - Using in-cluster MySQL
- [x] Security group allows EKS to RDS on port 3306 - N/A
- [x] Kubernetes secret created - Managed by deployKF

## Task 5: Install Kubeflow Pipelines v2 [COMPLETE via deployKF]

**Objective:** Deploy Kubeflow Pipelines to the EKS cluster.

### Commands

```bash
cd kubeflow-manifests

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.13.0/cert-manager.yaml
kubectl wait --for=condition=Available deployment --all -n cert-manager --timeout=300s

# Install Kubeflow Pipelines
kubectl apply -k apps/pipeline/upstream/env/aws
kubectl wait --for=condition=Available deployment --all -n kubeflow --timeout=600s
```

### Verification

```bash
kubectl get pods -n kubeflow | grep ml-pipeline
# Expected pods (all Running):
# - ml-pipeline-*
# - ml-pipeline-ui-*
# - ml-pipeline-persistenceagent-*
# - ml-pipeline-scheduledworkflow-*
# - workflow-controller-*
```

- [x] All pipeline pods in Running state
- [x] ml-pipeline service available
- [x] ml-pipeline-ui service available

## Task 6: Install Katib for Hyperparameter Tuning [COMPLETE via deployKF]

**Objective:** Deploy Katib for automated hyperparameter optimization.

### Commands

```bash
kubectl apply -k apps/katib/upstream/installs/katib-standalone
kubectl wait --for=condition=Available deployment --all -n kubeflow --timeout=300s
```

### Verification

```bash
kubectl get pods -n kubeflow | grep katib
kubectl get crd | grep katib
# Expected CRDs: experiments.kubeflow.org, suggestions.kubeflow.org, trials.kubeflow.org
```

- [x] Katib controller pod running
- [x] Katib UI pod running
- [x] Katib CRDs installed

## Task 7: Configure IAM Roles for Pipeline Pods (IRSA) [COMPLETE]

**Objective:** Set up IAM Roles for Service Accounts so pipeline pods can access AWS services.

### Create IAM Policy

Create `kubeflow-pipeline-policy.json`:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
      "Resource": ["arn:aws:s3:::ml-on-containers", "arn:aws:s3:::ml-on-containers/*"]
    },
    {
      "Effect": "Allow",
      "Action": ["ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage", "ecr:BatchCheckLayerAvailability", "ecr:GetAuthorizationToken"],
      "Resource": "*"
    }
  ]
}
```

### Create IAM Role

```bash
OIDC_PROVIDER=$(aws eks describe-cluster --name $CLUSTER_NAME \
  --query "cluster.identity.oidc.issuer" --output text | sed 's|https://||')

cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {"Federated": "arn:aws:iam::${AWS_ACCOUNT_ID}:oidc-provider/${OIDC_PROVIDER}"},
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "${OIDC_PROVIDER}:aud": "sts.amazonaws.com",
        "${OIDC_PROVIDER}:sub": "system:serviceaccount:kubeflow:pipeline-runner"
      }
    }
  }]
}
EOF

aws iam create-role --role-name KubeflowPipelineRole --assume-role-policy-document file://trust-policy.json
aws iam put-role-policy --role-name KubeflowPipelineRole --policy-name KubeflowPipelinePolicy --policy-document file://kubeflow-pipeline-policy.json
```

### Annotate Service Account

```bash
kubectl annotate serviceaccount pipeline-runner -n kubeflow \
  eks.amazonaws.com/role-arn=arn:aws:iam::${AWS_ACCOUNT_ID}:role/KubeflowPipelineRole --overwrite
```

### Verification

```bash
kubectl get sa pipeline-runner -n kubeflow -o yaml | grep eks.amazonaws.com
kubectl run test-s3-access --rm -it --image=amazon/aws-cli --serviceaccount=pipeline-runner -n kubeflow -- s3 ls s3://ml-on-containers/
```

- [x] IAM role created with correct trust policy
- [x] Service account annotated with role ARN
- [x] S3 access works from pod

## Task 8: Verify Installation [COMPLETE]

**Objective:** Confirm Kubeflow is fully operational.

### Access Kubeflow UI

```bash
kubectl port-forward svc/ml-pipeline-ui -n kubeflow 8080:80 &
# Access: http://localhost:8080
```

### Run Test Pipeline

```python
from kfp import dsl, compiler

@dsl.component(base_image='python:3.12-slim')
def hello_world(message: str) -> str:
    print(f"Hello from Kubeflow: {message}")
    return f"Processed: {message}"

@dsl.pipeline(name='installation-test-pipeline')
def test_pipeline(msg: str = 'Installation successful!'):
    hello_world(message=msg)

if __name__ == '__main__':
    compiler.Compiler().compile(test_pipeline, 'test_pipeline.yaml')
```

### Check All Components

```bash
echo "=== Kubeflow Pipelines ==="
kubectl get pods -n kubeflow -l app=ml-pipeline

echo "=== Katib ==="
kubectl get pods -n kubeflow -l katib.kubeflow.org/component

echo "=== Check for problems ==="
kubectl get pods -n kubeflow --field-selector=status.phase!=Running
```

### Verification Checklist

- [x] Kubeflow UI accessible via nginx-proxy port-forward
- [x] Test pipeline completes successfully
- [x] No pods in Error/CrashLoopBackOff state
- [x] Metrics visible in UI after pipeline run

## Task 9: Integration with Existing ArgoCD [COMPLETE]

**Objective:** Configure ArgoCD to manage Kubeflow components for GitOps.

### Create ArgoCD Application

Create `infra/manifests/argocd/kubeflow-app.yaml`:

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: kubeflow-pipelines
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/aws-samples/sample-financial-fraud-detection-with-nvidia
    targetRevision: main
    path: infra/manifests/kubeflow
  destination:
    server: https://kubernetes.default.svc
    namespace: kubeflow
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
      - CreateNamespace=true
```

### Apply and Verify

```bash
kubectl apply -f infra/manifests/argocd/kubeflow-app.yaml
kubectl get application kubeflow-pipelines -n argocd
```

### Verification

- [x] ArgoCD Application created
- [x] Application synced successfully
- [x] Kubeflow managed via GitOps

## Post-Installation Summary

| Component | Namespace | Verification Command |
|-----------|-----------|---------------------|
| Kubeflow Pipelines | kubeflow | `kubectl get pods -n kubeflow -l app=ml-pipeline` |
| Workflow Controller | kubeflow | `kubectl get pods -n kubeflow -l app=workflow-controller` |
| Katib | kubeflow | `kubectl get pods -n kubeflow -l katib.kubeflow.org/component` |

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Pods stuck in Pending | No nodes available | Check Karpenter logs, verify GPU node pool |
| S3 access denied | IRSA not configured | Verify service account annotations |
| UI not accessible | Service not exposed | Use port-forward or configure Ingress |
| Pipeline stuck | Workflow controller issue | Check `kubectl logs -n kubeflow deployment/workflow-controller` |

## Next Document

Proceed to [03-pipeline-components.md](./03-pipeline-components.md) for pipeline component specifications.
