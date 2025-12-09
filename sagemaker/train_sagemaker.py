#!/usr/bin/env python3
"""
SageMaker-compatible training script for Committor Model.
Handles SageMaker environment variables and S3 data loading.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Dataset
import argparse
import h5py
import numpy as np

# ==============================================================================
# Dataset
# ==============================================================================
class BackboneTokenDataset(Dataset):
    """Dataset for tokenized protein structures."""
    
    def __init__(self, h5_path, max_length=256):
        self.h5_path = h5_path
        self.max_length = max_length
        
        with h5py.File(h5_path, 'r') as f:
            self.length = len(f['labels'])
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            tokens = f['structure_tokens'][idx]
            label = f['labels'][idx]
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        elif len(tokens) < self.max_length:
            tokens = np.pad(tokens, (0, self.max_length - len(tokens)), constant_values=-1)
        
        # Replace padding (-1) with vocab padding token
        tokens = np.where(tokens == -1, 4095, tokens)
        
        return {
            'structure_tokens': torch.from_numpy(tokens).long(),
            'labels': torch.tensor(label, dtype=torch.float32)
        }

# ==============================================================================
# Model
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


class CommittorModel(nn.Module):
    """Transformer-based committor model."""
    
    def __init__(self, vocab_size=4096, embed_dim=256, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.structure_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=4095)
        self.pos_encoding = nn.Parameter(torch.randn(512, embed_dim) * 0.01)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=hidden_dim,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
        self.prediction_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                
    def forward(self, structure_tokens):
        embedded = self.structure_embedding(structure_tokens)
        embedded = embedded + self.pos_encoding[:structure_tokens.size(1)].unsqueeze(0)
        transformed = self.transformer(embedded)
        pooled = torch.mean(transformed, dim=1)
        logits = self.prediction_head(pooled).squeeze()
        return logits / self.temperature

# ==============================================================================
# SageMaker Environment
# ==============================================================================
def get_sagemaker_env():
    """Get SageMaker environment variables or use defaults."""
    env = {
        'model_dir': os.environ.get('SM_MODEL_DIR', './model'),
        'train_dir': os.environ.get('SM_CHANNEL_TRAINING', './data'),
        'output_dir': os.environ.get('SM_OUTPUT_DATA_DIR', './output'),
        'num_gpus': int(os.environ.get('SM_NUM_GPUS', torch.cuda.device_count())),
        'hosts': json.loads(os.environ.get('SM_HOSTS', '["localhost"]')),
        'current_host': os.environ.get('SM_CURRENT_HOST', 'localhost'),
    }
    return env


def setup_distributed():
    """Setup distributed training (works for both local and SageMaker)."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return local_rank, world_size, rank
    else:
        return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()

# ==============================================================================
# Training
# ==============================================================================
def train(args):
    """Main training function."""
    env = get_sagemaker_env()
    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    
    if rank == 0:
        print(f"=== Committor Model Training ===")
        print(f"Device: {device}")
        print(f"World size: {world_size}")
        print(f"SageMaker env: {env}")
    
    # Find data file
    data_path = args.data_path
    if not os.path.exists(data_path):
        # Try SageMaker channel
        data_path = os.path.join(env['train_dir'], os.path.basename(args.data_path))
    if not os.path.exists(data_path):
        # Try finding any h5 file
        for f in os.listdir(env['train_dir']):
            if f.endswith('.h5'):
                data_path = os.path.join(env['train_dir'], f)
                break
    
    if rank == 0:
        print(f"Data path: {data_path}")
    
    # Create dataset and loader
    dataset = BackboneTokenDataset(data_path)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=4, 
        pin_memory=True
    )
    
    # Create model
    model = CommittorModel().to(device)
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Dataset: {len(dataset)} samples")
        print(f"Model params: {n_params:,}")
    
    # Training setup
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.learning_rate * 10,
        steps_per_epoch=len(loader), epochs=args.epochs
    )
    
    best_loss = float('inf')
    checkpoint_dir = env['model_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(args.epochs):
        if sampler:
            sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(loader):
            structure_tokens = batch['structure_tokens'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            logits = model(structure_tokens)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        
        if rank == 0 and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}: Loss={avg_loss:.4f}")
        
        # Save best model
        if rank == 0 and avg_loss < best_loss:
            best_loss = avg_loss
            state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
            torch.save(state_dict, os.path.join(checkpoint_dir, 'best_model.pt'))
            print(f"  ✓ Best model saved (loss={best_loss:.4f})")
    
    # Save final model
    if rank == 0:
        state_dict = model.module.state_dict() if world_size > 1 else model.state_dict()
        torch.save(state_dict, os.path.join(checkpoint_dir, 'model.pt'))
        print(f"Training complete! Best loss: {best_loss:.4f}")
    
    cleanup_distributed()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='path_atlas_tokenized_esm3.h5')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--learning-rate', type=float, default=0.0001)
    
    # SageMaker passes hyperparameters
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))
    
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
