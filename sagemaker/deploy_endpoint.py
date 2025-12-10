#!/usr/bin/env python3
"""
Deploy Jinja Committor Model to SageMaker Endpoint using boto3
"""
import boto3
import os
import tarfile
import time
import argparse


def create_model_archive(model_path: str, output_path: str = "model.tar.gz"):
    """Package model for SageMaker deployment."""
    print("Creating model archive...")
    
    # Create inference code
    inference_code = '''#!/usr/bin/env python3
import torch
import json
import sys
sys.path.insert(0, '/opt/ml/model/code')

def model_fn(model_dir):
    from model import CommittorModel
    model = CommittorModel()
    model.load_state_dict(torch.load(f"{model_dir}/model.pt", map_location="cpu"))
    model.eval()
    return model

def input_fn(request_body, request_content_type):
    if request_content_type == "application/json":
        data = json.loads(request_body)
        return torch.tensor(data["tokens"]).long()
    raise ValueError(f"Unsupported: {request_content_type}")

def predict_fn(input_data, model):
    with torch.no_grad():
        logits = model(input_data)
        probs = torch.sigmoid(logits)
    return probs.numpy().tolist()

def output_fn(prediction, accept):
    return json.dumps({"committor_values": prediction})
'''
    
    # Model definition
    model_code = '''
import torch
import torch.nn as nn

class CommittorModel(nn.Module):
    def __init__(self, vocab_size=4096, embed_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.structure_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=4095)
        self.pos_encoding = nn.Parameter(torch.randn(512, embed_dim) * 0.01)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim,
            dropout=dropout, activation="gelu", batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, structure_tokens):
        embedded = self.structure_embedding(structure_tokens)
        embedded = embedded + self.pos_encoding[:structure_tokens.size(1)].unsqueeze(0)
        transformed = self.transformer(embedded)
        pooled = torch.mean(transformed, dim=1)
        logits = self.prediction_head(pooled).squeeze()
        return logits / self.temperature
'''
    
    os.makedirs("/tmp/model_package", exist_ok=True)
    os.makedirs("/tmp/model_package/code", exist_ok=True)
    
    with open("/tmp/model_package/code/inference.py", "w") as f:
        f.write(inference_code)
    with open("/tmp/model_package/code/model.py", "w") as f:
        f.write(model_code)
    
    # Copy model
    import shutil
    shutil.copy(model_path, "/tmp/model_package/model.pt")
    
    # Create tar.gz
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add("/tmp/model_package/model.pt", arcname="model.pt")
        tar.add("/tmp/model_package/code", arcname="code")
    
    print(f"✓ Created {output_path}")
    return output_path


def deploy_to_sagemaker(model_archive: str, endpoint_name: str = "jinja-committor"):
    """Deploy model to SageMaker using boto3."""
    
    session = boto3.Session()
    region = session.region_name
    account = boto3.client("sts").get_caller_identity()["Account"]
    
    s3 = boto3.client("s3")
    sm = boto3.client("sagemaker")
    
    bucket = f"sagemaker-{region}-{account}"
    model_key = f"jinja-model/{os.path.basename(model_archive)}"
    
    # Create bucket if needed
    try:
        s3.head_bucket(Bucket=bucket)
    except:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(Bucket=bucket, CreateBucketConfiguration={"LocationConstraint": region})
    
    # Upload model
    print(f"Uploading to s3://{bucket}/{model_key}...")
    s3.upload_file(model_archive, bucket, model_key)
    print("✓ Uploaded")
    
    model_data = f"s3://{bucket}/{model_key}"
    role_arn = f"arn:aws:iam::{account}:role/service-role/AmazonSageMakerServiceCatalogProductsExecutionRole"
    image_uri = f"763104351884.dkr.ecr.{region}.amazonaws.com/pytorch-inference:2.0.0-gpu-py310"
    
    model_name = f"{endpoint_name}-model"
    config_name = f"{endpoint_name}-config"
    
    # Delete existing if present
    for name in [endpoint_name]:
        try:
            sm.delete_endpoint(EndpointName=name)
            print(f"Deleted existing endpoint: {name}")
            time.sleep(10)
        except: pass
    for name in [config_name]:
        try:
            sm.delete_endpoint_config(EndpointConfigName=name)
        except: pass
    for name in [model_name]:
        try:
            sm.delete_model(ModelName=name)
        except: pass
    
    # Create model
    print("Creating SageMaker model...")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_data,
            "Environment": {"SAGEMAKER_PROGRAM": "inference.py"}
        },
        ExecutionRoleArn=role_arn
    )
    print(f"✓ Model: {model_name}")
    
    # Create endpoint config
    print("Creating endpoint config...")
    sm.create_endpoint_config(
        EndpointConfigName=config_name,
        ProductionVariants=[{
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InstanceType": "ml.g4dn.xlarge",
            "InitialInstanceCount": 1
        }]
    )
    print(f"✓ Config: {config_name}")
    
    # Create endpoint
    print("Creating endpoint (this takes 5-10 minutes)...")
    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )
    
    # Wait for deployment
    print("Waiting for endpoint to be ready...")
    waiter = sm.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    
    print(f"\n✓ Endpoint deployed: {endpoint_name}")
    print(f"  Region: {region}")
    print(f"  Invoke with: aws sagemaker-runtime invoke-endpoint --endpoint-name {endpoint_name} ...")
    
    return endpoint_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="checkpoints/best_model.pt")
    parser.add_argument("--endpoint-name", default="jinja-committor")
    parser.add_argument("--package-only", action="store_true")
    args = parser.parse_args()
    
    archive = create_model_archive(args.model)
    
    if not args.package_only:
        deploy_to_sagemaker(archive, args.endpoint_name)
    else:
        print("Package created. Run without --package-only to deploy.")


if __name__ == "__main__":
    main()
