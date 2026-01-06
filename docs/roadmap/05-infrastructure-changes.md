# 05 - Infrastructure Changes for Kubeflow Migration

**STATUS: COMPLETE**

This document details all infrastructure modifications required to support Kubeflow on the existing EKS cluster. Each section represents a discrete PR.

## Current Infrastructure Summary

The existing CDK deployment provisions:

- **VPC**: 3 AZs with NAT gateway, flow logs enabled
- **EKS**: v1.32 cluster via EKS Blueprints
- **Karpenter**: g4dn node pool (max 8 GPUs)
- **Addons**: GPU Operator v25.3.2, ArgoCD, ALB Controller, Secrets Store
- **Teams**: Triton namespace with S3 read-only ServiceAccount

## PR 1: Kubeflow IAM Role Stack [COMPLETE]

> Note: Actual implementation uses deployKF-managed service accounts with IRSA.

Create a new CDK stack for Kubeflow pipeline IAM permissions.

### File: `infra/lib/kubeflow-pipeline-role.ts`

```typescript
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface KubeflowPipelineRoleProps extends cdk.StackProps {
  modelBucketArn: string;
  clusterOidcProviderArn: string;
  clusterOidcIssuer: string;
}

export class KubeflowPipelineRoleStack extends cdk.Stack {
  public readonly pipelineRole: iam.Role;

  constructor(scope: Construct, id: string, props: KubeflowPipelineRoleProps) {
    super(scope, id, props);

    const oidcProvider = iam.OpenIdConnectProvider.fromOpenIdConnectProviderArn(
      this, 'OidcProvider', props.clusterOidcProviderArn
    );

    const federatedPrincipal = new iam.FederatedPrincipal(
      oidcProvider.openIdConnectProviderArn,
      {
        'StringEquals': {
          [`${props.clusterOidcIssuer}:aud`]: 'sts.amazonaws.com',
        },
        'StringLike': {
          [`${props.clusterOidcIssuer}:sub`]: 'system:serviceaccount:kubeflow:*',
        },
      },
      'sts:AssumeRoleWithWebIdentity'
    );

    this.pipelineRole = new iam.Role(this, 'KubeflowPipelineRole', {
      roleName: 'KubeflowPipelineExecutionRole',
      description: 'IAM role for Kubeflow pipeline pods via IRSA',
      assumedBy: federatedPrincipal,
      maxSessionDuration: cdk.Duration.hours(12),
    });

    // S3 access
    this.pipelineRole.addToPolicy(new iam.PolicyStatement({
      sid: 'S3ArtifactAccess',
      effect: iam.Effect.ALLOW,
      actions: ['s3:GetObject', 's3:PutObject', 's3:DeleteObject', 's3:ListBucket'],
      resources: [props.modelBucketArn, `${props.modelBucketArn}/*`],
    }));

    // ECR access
    this.pipelineRole.addToPolicy(new iam.PolicyStatement({
      sid: 'ECRPullAccess',
      effect: iam.Effect.ALLOW,
      actions: ['ecr:GetAuthorizationToken', 'ecr:BatchCheckLayerAvailability',
                'ecr:GetDownloadUrlForLayer', 'ecr:BatchGetImage'],
      resources: ['*'],
    }));

    new cdk.CfnOutput(this, 'PipelineRoleArn', {
      value: this.pipelineRole.roleArn,
      exportName: 'KubeflowPipelineRoleArn',
    });
  }
}
```

## PR 2: Kubeflow Namespace and RBAC [COMPLETE via deployKF]

> Note: deployKF manages these resources automatically.

### File: `infra/manifests/kubeflow/namespace.yaml`

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: kubeflow
  labels:
    app.kubernetes.io/part-of: kubeflow
    pod-security.kubernetes.io/enforce: baseline
```

### File: `infra/manifests/kubeflow/rbac.yaml`

```yaml
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: kubeflow-pipeline-runner
  namespace: kubeflow
  annotations:
    eks.amazonaws.com/role-arn: arn:aws:iam::ACCOUNT_ID:role/KubeflowPipelineExecutionRole
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: kubeflow-pipeline-runner
rules:
  - apiGroups: [""]
    resources: ["pods", "pods/log", "pods/status"]
    verbs: ["get", "list", "watch", "create", "delete"]
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["argoproj.io"]
    resources: ["workflows", "workflows/status"]
    verbs: ["get", "list", "watch", "create", "update", "patch", "delete"]
  - apiGroups: ["batch"]
    resources: ["jobs", "jobs/status"]
    verbs: ["get", "list", "watch", "create", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: kubeflow-pipeline-runner
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: kubeflow-pipeline-runner
subjects:
  - kind: ServiceAccount
    name: kubeflow-pipeline-runner
    namespace: kubeflow
```

### File: `infra/manifests/kubeflow/network-policy.yaml`

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: allow-kubeflow-internal
  namespace: kubeflow
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              app.kubernetes.io/part-of: kubeflow
  egress:
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
```

## PR 3: Karpenter Node Pools [COMPLETE]

GPU node pools are configured in the CDK blueprint.

Add CPU node pool for preprocessing workloads. Update `infra/lib/nvidia-fraud-detection-blueprint.ts`:

### CPU Node Pool (Preprocessing)

```typescript
const cpuNodePoolSpec: blueprints.NodePoolV1Spec = {
  labels: {
    "node-type": "cpu",
    "workload": "ml-preprocessing",
  },
  taints: [],
  requirements: [
    { key: "karpenter.sh/capacity-type", operator: "In", values: ["spot", "on-demand"] },
    { key: "node.kubernetes.io/instance-type", operator: "In",
      values: ["m5.xlarge", "m5.2xlarge", "m5.4xlarge"] },
    { key: "kubernetes.io/arch", operator: "In", values: ["amd64"] },
  ],
  expireAfter: "4h",
  disruption: {
    consolidationPolicy: "WhenEmptyOrUnderutilized",
    consolidateAfter: "1m"
  },
  limits: { cpu: 128, memory: "512Gi" },
  weight: 50
};
```

### GPU Training Node Pool (Existing - No Changes)

The existing g4dn pool is suitable for training. Just verify labels:

```typescript
labels: {
  "node-type": "gpu",
  "nvidia.com/gpu": "true",
  "workload": "ml-training"
}
```

## PR 4: ArgoCD Application for Kubeflow [COMPLETE via deployKF]

> Note: deployKF uses its own ArgoCD app-of-apps pattern.

### File: `infra/manifests/argocd/kubeflow-app.yaml`

```yaml
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: kubeflow-pipelines
  namespace: argocd
  annotations:
    argocd.argoproj.io/sync-wave: "1"
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
      - ServerSideApply=true
```

## PR 5: S3 Bucket Policy Update [COMPLETE]

S3 access configured via IRSA on service accounts.

Update the model bucket to allow Kubeflow artifact storage:

### File: `infra/lib/s3-bucket-policy.ts`

```typescript
bucket.addToResourcePolicy(new iam.PolicyStatement({
  sid: 'KubeflowArtifactAccess',
  effect: iam.Effect.ALLOW,
  principals: [new iam.ArnPrincipal(kubeflowPipelineRoleArn)],
  actions: ['s3:GetObject', 's3:PutObject', 's3:DeleteObject'],
  resources: [`${bucket.bucketArn}/kubeflow-artifacts/*`],
}));
```

## PR 6: Kubeflow Kustomization [SKIPPED - Using deployKF]

> Note: deployKF handles kustomization internally.

### File: `infra/manifests/kubeflow/kustomization.yaml`

```yaml
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

namespace: kubeflow

resources:
  - namespace.yaml
  - rbac.yaml
  - network-policy.yaml

configMapGenerator:
  - name: kubeflow-config
    literals:
      - S3_BUCKET=ml-on-containers
      - ARTIFACT_PREFIX=kubeflow-artifacts

images:
  - name: gcr.io/ml-pipeline/api-server
    newTag: 2.0.5
```

## Deployment Order

```
PR 1: Kubeflow IAM Role ──┐
                          ├──► PR 2: Namespace/RBAC ──► PR 4: ArgoCD App
PR 3: Node Pools ─────────┘
                                        │
                                        v
                               PR 5: S3 Policy
                                        │
                                        v
                               PR 6: Kustomization
```

Phase 1 and Phase 3 can run in parallel. All PRs should be merged before Task 5 (Install Kubeflow).

## Additional Infrastructure Added

These were added beyond the original plan:

### Custom Triton Image (infra/lib/triton-image-repo.ts)

- ECR repository for custom Triton inference image
- CodeBuild project builds image from `triton/Dockerfile`
- Lambda custom resource triggers build on CDK deploy
- Image includes PyTorch, torch_geometric, XGBoost, Captum

### nginx Proxy (infra/manifests/nginx-proxy.yaml)

- TCP proxy pod for stable port-forwarding to Istio gateway
- Resolves connection drops during development

## Validation Checklist [ALL PASSED]

| PR | Validation Command | Status |
|----|-------------------|
| 1 | `aws iam get-role --role-name KubeflowPipelineExecutionRole` |
| 2 | `kubectl get ns kubeflow && kubectl get sa -n kubeflow` |
| 3 | `kubectl get nodepools` |
| 4 | `kubectl get applications -n argocd` |
| 5 | `aws s3api get-bucket-policy --bucket ml-on-containers` |
| 6 | `kubectl kustomize infra/manifests/kubeflow/` |

## CDK Deployment Commands

```bash
cd infra

# Deploy IAM role stack
npx cdk deploy NvidiaFraudDetectionKubeflowRole

# Update main stack with new node pool
npx cdk deploy NvidiaFraudDetectionBlueprint

# Verify outputs
aws cloudformation describe-stacks \
  --stack-name NvidiaFraudDetectionKubeflowRole \
  --query 'Stacks[0].Outputs'
```

## Notes for AI Agents

1. Extract OIDC provider ARN from existing cluster stack output
2. Replace `ACCOUNT_ID` placeholders via CDK context or environment
3. Node pool weights determine scheduling priority (higher = preferred)
4. ArgoCD sync waves ensure ordered deployment
5. Network policies assume no Istio service mesh

## Summary

All infrastructure changes have been deployed. The EKS cluster runs Kubeflow via deployKF, with custom Triton serving, GPU node pools via Karpenter, and GitOps via ArgoCD.
