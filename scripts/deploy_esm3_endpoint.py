#!/usr/bin/env python3
"""
Deploy ESM3-open model from SageMaker JumpStart as a real-time endpoint.
This allows tokenizing protein structures without needing local gated model access.
"""

import boto3
import json
import time
import argparse
from datetime import datetime


def get_jumpstart_model_id():
    """Get the ESM3-open model ID from JumpStart."""
    # ESM3-open is available in JumpStart
    # Model ID format may vary, this is the expected pattern
    return "huggingface-embedding-esm3-open-small"


def deploy_esm3_endpoint(
    endpoint_name: str = None,
    instance_type: str = "ml.g5.xlarge",
    region: str = None
):
    """
    Deploy ESM3-open model to a SageMaker endpoint.
    
    Args:
        endpoint_name: Name for the endpoint (auto-generated if None)
        instance_type: SageMaker instance type
        region: AWS region
    
    Returns:
        endpoint_name: Name of the deployed endpoint
    """
    if region is None:
        region = boto3.Session().region_name or "us-east-1"
    
    if endpoint_name is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        endpoint_name = f"esm3-open-{timestamp}"
    
    print(f"=== Deploying ESM3-open Endpoint ===")
    print(f"Endpoint Name: {endpoint_name}")
    print(f"Instance Type: {instance_type}")
    print(f"Region: {region}")
    
    # Initialize SageMaker client
    sm_client = boto3.client("sagemaker", region_name=region)
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
    
    # Check if endpoint already exists
    try:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status == "InService":
            print(f"✓ Endpoint {endpoint_name} already exists and is InService")
            return endpoint_name
        elif status in ["Creating", "Updating"]:
            print(f"⏳ Endpoint {endpoint_name} is {status}, waiting...")
            wait_for_endpoint(sm_client, endpoint_name)
            return endpoint_name
        else:
            print(f"⚠ Endpoint {endpoint_name} is in state {status}, recreating...")
            sm_client.delete_endpoint(EndpointName=endpoint_name)
            time.sleep(30)
    except sm_client.exceptions.ClientError:
        pass  # Endpoint doesn't exist, we'll create it
    
    # Try JumpStart first
    try:
        from sagemaker.jumpstart.model import JumpStartModel
        
        print("Deploying via SageMaker JumpStart...")
        model = JumpStartModel(
            model_id="huggingface-embedding-esm3-open-small",
            role=get_sagemaker_role()
        )
        
        predictor = model.deploy(
            initial_instance_count=1,
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            wait=False
        )
        
        print(f"⏳ Deployment initiated, waiting for endpoint to be InService...")
        wait_for_endpoint(sm_client, endpoint_name)
        print(f"✓ Endpoint {endpoint_name} is now InService")
        return endpoint_name
        
    except Exception as e:
        print(f"JumpStart deployment failed: {e}")
        print("Attempting Marketplace model deployment...")
        return deploy_from_marketplace(endpoint_name, instance_type, region)


def deploy_from_marketplace(endpoint_name: str, instance_type: str, region: str):
    """
    Deploy ESM3 from AWS Marketplace as fallback.
    """
    sm_client = boto3.client("sagemaker", region_name=region)
    
    # Note: User needs to subscribe to the model package first
    # This is a placeholder for the marketplace model ARN
    print("⚠ Marketplace deployment requires manual subscription first.")
    print("Please subscribe to ESM3-open in AWS Marketplace and try again.")
    print("Marketplace URL: https://aws.amazon.com/marketplace/pp/prodview-esm3")
    return None


def get_sagemaker_role():
    """Get the SageMaker execution role."""
    iam = boto3.client("iam")
    
    # Try to find existing SageMaker role
    try:
        response = iam.get_role(RoleName="SageMakerExecutionRole")
        return response["Role"]["Arn"]
    except iam.exceptions.NoSuchEntityException:
        pass
    
    # Use the current instance role if available
    try:
        sts = boto3.client("sts")
        response = sts.get_caller_identity()
        role_arn = response["Arn"]
        # Convert assumed-role ARN to role ARN
        if ":assumed-role/" in role_arn:
            parts = role_arn.split("/")
            role_name = parts[1]
            account = response["Account"]
            return f"arn:aws:iam::{account}:role/{role_name}"
        return role_arn
    except Exception as e:
        print(f"Error getting role: {e}")
        raise ValueError("Could not determine SageMaker execution role")


def wait_for_endpoint(sm_client, endpoint_name, timeout=1800):
    """Wait for endpoint to be InService."""
    start_time = time.time()
    while True:
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        
        if status == "InService":
            return True
        elif status == "Failed":
            raise Exception(f"Endpoint creation failed: {response.get('FailureReason', 'Unknown')}")
        elif time.time() - start_time > timeout:
            raise TimeoutError(f"Endpoint creation timed out after {timeout}s")
        
        print(f"  Status: {status}...")
        time.sleep(30)


def test_endpoint(endpoint_name: str, region: str = None):
    """Test the deployed endpoint with a sample sequence."""
    if region is None:
        region = boto3.Session().region_name or "us-east-1"
    
    sm_runtime = boto3.client("sagemaker-runtime", region_name=region)
    
    # Sample protein sequence
    test_sequence = "MKTAYIAKQRQISFVK"
    
    print(f"\n=== Testing Endpoint ===")
    print(f"Sequence: {test_sequence}")
    
    try:
        response = sm_runtime.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="application/json",
            Body=json.dumps({"inputs": test_sequence})
        )
        
        result = json.loads(response["Body"].read().decode())
        print(f"✓ Response received: {type(result)}")
        if isinstance(result, list):
            print(f"  Embedding dimension: {len(result[0]) if result else 'N/A'}")
        return True
        
    except Exception as e:
        print(f"✗ Error testing endpoint: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Deploy ESM3-open endpoint")
    parser.add_argument("--endpoint-name", default=None, help="Endpoint name")
    parser.add_argument("--instance-type", default="ml.g5.xlarge", help="Instance type")
    parser.add_argument("--region", default=None, help="AWS region")
    parser.add_argument("--test", action="store_true", help="Test endpoint after deployment")
    parser.add_argument("--test-only", type=str, help="Only test an existing endpoint")
    
    args = parser.parse_args()
    
    if args.test_only:
        test_endpoint(args.test_only, args.region)
        return
    
    endpoint_name = deploy_esm3_endpoint(
        endpoint_name=args.endpoint_name,
        instance_type=args.instance_type,
        region=args.region
    )
    
    if endpoint_name and args.test:
        test_endpoint(endpoint_name, args.region)
    
    print(f"\n=== Deployment Complete ===")
    print(f"Endpoint: {endpoint_name}")
    print(f"\nTo use in tokenization, set:")
    print(f"  export ESM3_ENDPOINT={endpoint_name}")


if __name__ == "__main__":
    main()
