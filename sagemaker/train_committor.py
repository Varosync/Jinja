#!/usr/bin/env python3
"""
SageMaker Training Job launcher for Committor Model.
Uses PyTorch Estimator with BioNeMo-compatible environment.
"""

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
import argparse
from datetime import datetime
import os


def get_sagemaker_role():
    """Get the SageMaker execution role."""
    sts = boto3.client("sts")
    response = sts.get_caller_identity()
    role_arn = response["Arn"]
    
    if ":assumed-role/" in role_arn:
        parts = role_arn.split("/")
        role_name = parts[1]
        account = response["Account"]
        return f"arn:aws:iam::{account}:role/{role_name}"
    return role_arn


def launch_training_job(
    s3_data_path: str,
    s3_output_path: str = None,
    instance_type: str = "ml.g5.xlarge",
    instance_count: int = 1,
    epochs: int = 150,
    batch_size: int = 64,
    learning_rate: float = 0.0001,
    job_name: str = None
):
    """
    Launch a SageMaker training job for the Committor model.
    
    Args:
        s3_data_path: S3 URI to training data (HDF5 file)
        s3_output_path: S3 URI for model artifacts output
        instance_type: SageMaker instance type
        instance_count: Number of training instances
        epochs: Number of training epochs
        batch_size: Training batch size
        learning_rate: Learning rate
        job_name: Custom job name (auto-generated if None)
    
    Returns:
        Training job name
    """
    session = sagemaker.Session()
    role = get_sagemaker_role()
    
    if s3_output_path is None:
        bucket = session.default_bucket()
        s3_output_path = f"s3://{bucket}/jinja/output"
    
    if job_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        job_name = f"jinja-committor-{timestamp}"
    
    print(f"=== SageMaker Training Job ===")
    print(f"Job Name: {job_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Instance Count: {instance_count}")
    print(f"Data: {s3_data_path}")
    print(f"Output: {s3_output_path}")
    
    # Define hyperparameters
    hyperparameters = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "d_model": 256,
        "n_layers": 3,
        "n_heads": 8,
        "dropout": 0.3
    }
    
    # Create PyTorch Estimator
    estimator = PyTorch(
        entry_point="train_committor.py",
        source_dir="committor_training",
        role=role,
        instance_count=instance_count,
        instance_type=instance_type,
        framework_version="2.0.0",
        py_version="py310",
        hyperparameters=hyperparameters,
        output_path=s3_output_path,
        job_name=job_name,
        max_run=86400,  # 24 hours max
        distribution={
            "torch_distributed": {
                "enabled": True
            }
        } if instance_count > 1 else None
    )
    
    # Start training
    print("\n⏳ Starting training job...")
    estimator.fit({"training": s3_data_path}, wait=False)
    
    print(f"\n✓ Training job {job_name} submitted")
    print(f"\nTo monitor progress:")
    print(f"  aws sagemaker describe-training-job --training-job-name {job_name}")
    print(f"\nTo view logs:")
    print(f"  aws logs tail /aws/sagemaker/TrainingJobs --prefix {job_name}")
    
    return job_name


def check_job_status(job_name: str):
    """Check the status of a training job."""
    sm_client = boto3.client("sagemaker")
    
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    status = response["TrainingJobStatus"]
    
    print(f"Job: {job_name}")
    print(f"Status: {status}")
    
    if status == "Completed":
        print(f"Model Artifacts: {response['ModelArtifacts']['S3ModelArtifacts']}")
    elif status == "Failed":
        print(f"Failure Reason: {response.get('FailureReason', 'Unknown')}")
    
    return status


def main():
    parser = argparse.ArgumentParser(description="Launch SageMaker training job")
    parser.add_argument("--data", required=True, help="S3 URI to training data")
    parser.add_argument("--output", default=None, help="S3 URI for output")
    parser.add_argument("--instance-type", default="ml.g5.xlarge", help="Instance type")
    parser.add_argument("--instance-count", type=int, default=1, help="Instance count")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--job-name", default=None, help="Custom job name")
    parser.add_argument("--status", type=str, help="Check status of existing job")
    
    args = parser.parse_args()
    
    if args.status:
        check_job_status(args.status)
        return
    
    launch_training_job(
        s3_data_path=args.data,
        s3_output_path=args.output,
        instance_type=args.instance_type,
        instance_count=args.instance_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        job_name=args.job_name
    )


if __name__ == "__main__":
    main()
