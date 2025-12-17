"""SageMaker training job management for fraud detection GNN model."""
import time
from dataclasses import dataclass
from typing import Optional

import boto3
from IPython.display import clear_output


@dataclass
class SageMakerTrainingConfig:
    """Configuration for SageMaker training job."""
    bucket_name: str
    sagemaker_training_role: str
    training_repo: str
    
    # Optional overrides
    job_name_prefix: str = "fraud-detection-gnn"
    instance_type: str = "ml.g5.xlarge"
    instance_count: int = 1
    volume_size_gb: int = 30
    max_runtime_seconds: int = 86400
    
    # CUDA compatibility version
    # Change to "cuda-compat-13-0" for CUDA 13 (requires Host Driver >= 580)
    # Use "cuda-compat-12-2" if you see "cudaErrorInsufficientDriver"
    cuda_compat_version: str = "cuda-compat-13-0"
    
    # SSM parameter name for model tracking
    ssm_parameter_name: str = "/triton/model"


class SageMakerTrainingJob:
    """Manages SageMaker training jobs for the fraud detection GNN model."""
    
    def __init__(self, config: SageMakerTrainingConfig):
        """
        Initialize the training job manager.
        
        Args:
            config: SageMakerTrainingConfig with all required settings.
        """
        self.config = config
        self.ssm_client = boto3.client("ssm")
        self.s3_client = boto3.client("s3")
        self.sagemaker_client = boto3.client("sagemaker")
        self.training_job_name: Optional[str] = None
        self.script_s3_uri: Optional[str] = None
    
    def _generate_job_name(self) -> str:
        """Generate a unique training job name with timestamp."""
        return f"{self.config.job_name_prefix}-{int(time.time())}"
    
    def _register_job_in_parameter_store(self) -> None:
        """Register the training job name in SSM Parameter Store."""
        self.ssm_client.put_parameter(
            Name=self.config.ssm_parameter_name,
            Value=self.training_job_name,
            Overwrite=True,
            Type="String"
        )
    
    def _generate_entrypoint_script(self) -> str:
        """Generate the custom entrypoint bash script."""
        return f"""#!/bin/bash
set -e  # Exit immediately if any command fails

echo "--- HOST DEBUG INFO ---"
cat /etc/os-release
echo "NVIDIA Driver Version on Host:"
nvidia-smi --query-gpu=driver_version --format=csv,noheader
echo "-----------------------"

echo "Starting custom setup..."

# A. Install Python dependencies
pip install --upgrade boto3 pandas

# B. Install System/CUDA packages
# We accept failure (|| true) on apt-get update in case of transient network issues,
# but the install must succeed.
apt-get update || true
apt-get install -y {self.config.cuda_compat_version}

# C. Configure Linker for Forward Compatibility
# The compat package installs libs to /usr/local/cuda/compat
export LD_LIBRARY_PATH=/usr/local/cuda/compat:$LD_LIBRARY_PATH

echo "Setup complete. Launching training..."

# D. Run the actual training command
# We use exec to replace the shell process with torchrun
exec torchrun --standalone --nnodes=1 --nproc_per_node=1 /app/main.py --config /opt/ml/input/data/config/training_config.json
"""
    
    def _upload_entrypoint_script(self) -> str:
        """
        Upload the entrypoint script to S3.
        
        Returns:
            S3 URI of the uploaded script.
        """
        script_content = self._generate_entrypoint_script()
        script_key = f"code/{self.training_job_name}/entrypoint.sh"
        
        self.s3_client.put_object(
            Body=script_content,
            Bucket=self.config.bucket_name,
            Key=script_key
        )
        
        return f"s3://{self.config.bucket_name}/{script_key}"
    
    def _create_training_job(self) -> dict:
        """
        Create the SageMaker training job.
        
        Returns:
            Response from SageMaker create_training_job API.
        """
        return self.sagemaker_client.create_training_job(
            TrainingJobName=self.training_job_name,
            RoleArn=self.config.sagemaker_training_role,
            AlgorithmSpecification={
                'TrainingImage': f"{self.config.training_repo}:latest",
                'TrainingInputMode': 'File',
                'ContainerEntrypoint': ['/bin/bash', '-c'],
                'ContainerArguments': [
                    f"pip install awscli && aws s3 cp {self.script_s3_uri} /tmp/run.sh && bash /tmp/run.sh"
                ]
            },
            InputDataConfig=[
                {
                    'ChannelName': 'data',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{self.config.bucket_name}/data/gnn/',
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'ContentType': 'application/x-directory',
                    'InputMode': 'File',
                    'CompressionType': 'None'
                },
                {
                    'ChannelName': 'config',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': f's3://{self.config.bucket_name}/config/',
                            'S3DataDistributionType': 'FullyReplicated',
                        }
                    },
                    'ContentType': 'application/x-directory',
                    'InputMode': 'File',
                    'CompressionType': 'None'
                }
            ],
            OutputDataConfig={
                'S3OutputPath': f's3://{self.config.bucket_name}/output/'
            },
            ResourceConfig={
                'InstanceType': self.config.instance_type,
                'InstanceCount': self.config.instance_count,
                'VolumeSizeInGB': self.config.volume_size_gb
            },
            StoppingCondition={
                'MaxRuntimeInSeconds': self.config.max_runtime_seconds
            }
        )
    
    def launch(self, verbose: bool = True) -> str:
        """
        Launch the complete training job workflow.
        
        This method:
        1. Generates a unique job name
        2. Registers it in SSM Parameter Store
        3. Uploads the custom entrypoint script to S3
        4. Creates the SageMaker training job
        
        Args:
            verbose: If True, print progress messages.
            
        Returns:
            The training job ARN.
        """
        # Step 1: Generate job name
        self.training_job_name = self._generate_job_name()
        if verbose:
            print(f"Training job name: {self.training_job_name}")
        
        # Step 2: Register in Parameter Store
        self._register_job_in_parameter_store()
        if verbose:
            print(f"Registered job name in SSM: {self.config.ssm_parameter_name}")
        
        # Step 3: Upload entrypoint script
        self.script_s3_uri = self._upload_entrypoint_script()
        if verbose:
            print(f"Uploaded custom entrypoint to {self.script_s3_uri}")
        
        # Step 4: Create training job
        response = self._create_training_job()
        training_job_arn = response['TrainingJobArn']
        if verbose:
            print(f"Training job started: {training_job_arn}")
        
        return training_job_arn
    
    def poll(self):
        return poll_training_status(self.training_job_name)


def poll_training_status(job_name):
    sagemaker_client = boto3.client('sagemaker')
    
    while True:
        response = sagemaker_client.describe_training_job(TrainingJobName=job_name)
        status = response['TrainingJobStatus']
        
        # Clear previous output
        clear_output(wait=True)
        
        # Print current status with timestamp
        print(f"Job: {job_name}")
        print(f"Status: {status}")
        print(f"Last checked: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if status in ['Completed', 'Failed', 'Stopped']:
            break
            
        # Wait before next check
        time.sleep(30)
    
    return status