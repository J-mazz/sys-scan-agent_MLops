#!/bin/bash
# Launch SageMaker training job for security analysis model
# Uses ml.g6.2xlarge instance (L4 GPU) to avoid EC2 quota issues

set -e

echo "🚀 Launching SageMaker training job for sys-scan security analysis model..."

# Your SageMaker role ARN (update this!)
ROLE_ARN="arn:aws:iam::123456789012:role/SageMakerRole"

# S3 bucket for data and checkpoints (update this!)
S3_BUCKET="your-sagemaker-bucket"
S3_DATA_PATH="s3://${S3_BUCKET}/training-data/"
S3_CHECKPOINT_PATH="s3://${S3_BUCKET}/checkpoints/"

echo "Using S3 bucket: ${S3_BUCKET}"
echo "Data path: ${S3_DATA_PATH}"
echo "Checkpoint path: ${S3_CHECKPOINT_PATH}"

# Upload training data to S3 (if not already uploaded)
echo "📤 Uploading training data to S3..."
aws s3 sync ./massive_datasets/ ${S3_DATA_PATH}massive_datasets/ --exclude "*" --include "*.json" || true

# Create and launch SageMaker training job
python3 -c "
import boto3
from sagemaker.pytorch import PyTorch
import time

# Initialize SageMaker client
sm_client = boto3.client('sagemaker')

# Create estimator
estimator = PyTorch(
    entry_point='train.py',
    source_dir='.',
    role='${ROLE_ARN}',
    instance_type='ml.g6.2xlarge',
    instance_count=1,
    framework_version='2.0.1',
    py_version='py310',
    volume_size=150,
    max_run=129600,  # 36 hours
    use_spot_instances=True,
    max_wait=14400,   # 4 hours wait for Spot
    checkpoint_s3_uri='${S3_CHECKPOINT_PATH}',
    hyperparameters={
        'epochs': 3,
        'learning_rate': 2e-4,
        'batch_size': 4,
        'beta_1': 0.9,
        'beta_2': 0.99,
        'weight_decay': 0.01,
    },
    environment={
        'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
    }
)

# Generate unique job name
job_name = f'sys-scan-mistral-training-{int(time.time())}'

print(f'Launching training job: {job_name}')
print(f'Instance type: ml.g6.2xlarge (L4 GPU)')
print(f'Using Spot instances for cost savings')

# Launch training job
estimator.fit(
    {
        'training': '${S3_DATA_PATH}',
        'validation': '${S3_DATA_PATH}'
    },
    job_name=job_name
)

print(f'✅ Training job {job_name} launched successfully!')
print(f'Monitor progress at: https://console.aws.amazon.com/sagemaker/home?#/jobs/{job_name}')
"

echo "🎉 SageMaker training job launched!"
echo "Monitor your training job in the AWS SageMaker console."
echo "Estimated cost: ~$2-3/hour with Spot instances"