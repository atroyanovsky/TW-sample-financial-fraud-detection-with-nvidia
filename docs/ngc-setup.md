# NGC Container Registry Setup

This guide covers how to configure your EKS cluster to pull private images from NVIDIA NGC (nvcr.io).

## Prerequisites

You need an NGC API key. Get one from [NGC](https://ngc.nvidia.com/):
1. Log in to NGC
2. Go to Setup > API Key
3. Generate a new key and save it securely

## Create the NGC Secret

Create a Kubernetes secret with your NGC credentials. The username is always `$oauthtoken` and the password is your API key.

```bash
# Set your NGC API key
export NGC_API_KEY="your-api-key-here"

# Create the secret in your namespace
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" \
  -n team-1
```

## Verify the Secret

```bash
kubectl get secret ngc-secret -n team-1 -o yaml
```

## Using NGC Images in Kubeflow Pipelines

For container components that use NGC images, you need to add the imagePullSecret. In KFP 2.x with kfp-kubernetes, you can patch the compiled YAML or add it programmatically.

### Option 1: Patch the Compiled YAML

After compiling your pipeline, add the imagePullSecrets to the kubernetes platform spec:

```yaml
platforms:
  kubernetes:
    deploymentSpec:
      executors:
        exec-run-nvidia-training-on-pvc:
          imagePullSecrets:
          - name: ngc-secret
          nodeSelector:
            labels:
              nvidia.com/gpu: 'true'
          pvcMount:
          - mountPath: /data
            ...
```

### Option 2: Create a Default Service Account

Create a service account with the imagePullSecret attached, then use it for all pods:

```bash
# Patch the default service account in your namespace
kubectl patch serviceaccount default -n team-1 \
  -p '{"imagePullSecrets": [{"name": "ngc-secret"}]}'
```

This automatically applies to all pods in the namespace that don't specify a different service account.

### Option 3: Namespace-wide ImagePullSecret (Recommended)

For deployKF/Kubeflow, patch the profile's default service account:

```bash
kubectl patch serviceaccount default-editor -n team-1 \
  -p '{"imagePullSecrets": [{"name": "ngc-secret"}]}'
```

## Testing

Test that the secret works by pulling an NGC image:

```bash
kubectl run ngc-test --rm -it \
  --image=nvcr.io/nvidia/pytorch:24.01-py3 \
  --overrides='{"spec":{"imagePullSecrets":[{"name":"ngc-secret"}]}}' \
  -n team-1 \
  -- nvidia-smi
```

## Using ECR Instead (Current Setup)

The current pipeline uses ECR (`123456789012.dkr.ecr.us-west-2.amazonaws.com/nvidia-training-repo:latest`) which doesn't require imagePullSecrets since the EKS nodes have IAM permissions to pull from ECR in the same account.

To switch back to NGC images, update `e2e_pipeline.py`:

```python
@dsl.container_component
def run_nvidia_training_on_pvc():
    return dsl.ContainerSpec(
        image="nvcr.io/nvidia/cugraph/financial-fraud-training:2.0.0",
        ...
    )
```

Then either patch the YAML or use the service account method above.

## Troubleshooting

### ImagePullBackOff with 401 Unauthorized
- Verify the secret exists: `kubectl get secret ngc-secret -n team-1`
- Check the secret is correctly formatted: `kubectl get secret ngc-secret -n team-1 -o jsonpath='{.data.\.dockerconfigjson}' | base64 -d`
- Ensure the API key is valid and not expired

### Pod Not Using the Secret
- Check if imagePullSecrets is in the pod spec: `kubectl get pod <pod-name> -n team-1 -o yaml | grep -A2 imagePullSecrets`
- If using service account method, verify: `kubectl get sa default -n team-1 -o yaml`
