#!/usr/bin/env python3
"""
Torchrun-compatible training script for BioNeMo committor model with backbone structure tokens.
This script is designed to work with torchrun for multi-GPU training.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import h5py
import numpy as np
import argparse
import json
from datetime import datetime

class BackboneTokenDataset(Dataset):
    """Dataset for backbone structure tokens compatible with distributed training."""
    
    def __init__(self, hdf5_file, max_samples=None, transform=None):
        """
        Args:
            hdf5_file (str): Path to the tokenized backbone HDF5 file
            max_samples (int, optional): Maximum number of samples to use
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.hdf5_file = hdf5_file
        self.transform = transform
        
        with h5py.File(hdf5_file, 'r') as f:
            self.length = len(f['labels'])
            if max_samples:
                self.length = min(self.length, max_samples)
            
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            # Load data
            label = f['labels'][idx]
            structure_tokens = f['structure_tokens'][idx]
            sequence = f['sequences'][idx]
            protein_name = f['protein_names'][idx]
            
            # Convert to tensors
            structure_tokens = torch.from_numpy(structure_tokens).long()
            label = torch.tensor(label).float()
            sequence_str = sequence.decode('utf-8') if isinstance(sequence, bytes) else str(sequence)
            protein_name_str = protein_name.decode('utf-8') if isinstance(protein_name, bytes) else str(protein_name)
            
            # Adjust token values: map -1 (padding) to 4095 (our padding index)
            structure_tokens = torch.where(structure_tokens == -1, 
                                         torch.tensor(4095), 
                                         structure_tokens)
            
            # Apply transforms if any
            if self.transform:
                structure_tokens = self.transform(structure_tokens)
            
        return {
            'structure_tokens': structure_tokens,
            'labels': label,
            'sequences': sequence_str,
            'protein_names': protein_name_str
        }

class BioNeMoBackboneCommittorModel(nn.Module):
    """Enhanced committor model that uses backbone structure tokens as primary input."""
    
    def __init__(self, vocab_size=4096, embed_dim=256, hidden_dim=512, dropout=0.1):
        """
        Args:
            vocab_size (int): Size of the structure token vocabulary
            embed_dim (int): Dimension of the embedding layer
            hidden_dim (int): Dimension of hidden layers
            dropout (float): Dropout rate
        """
        super(BioNeMoBackboneCommittorModel, self).__init__()
        
        # Structure token embedding
        self.structure_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=4095)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(512, embed_dim))
        
        # Transformer-like layers for feature extraction
        self.transformer_layers = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism to focus on important regions
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=8, dropout=dropout, batch_first=True)
        
        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                
    def forward(self, structure_tokens):
        """
        Forward pass through the model.
        
        Args:
            structure_tokens (torch.Tensor): Tensor of shape (batch_size, seq_len)
            
        Returns:
            torch.Tensor: Predictions of shape (batch_size,)
        """
        batch_size, seq_len = structure_tokens.shape
        
        # Embed structure tokens
        embedded = self.structure_embedding(structure_tokens)  # (batch, seq_len, embed_dim)
        
        # Add positional encoding
        embedded = embedded + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Apply attention mechanism
        attended, _ = self.attention(embedded, embedded, embedded)
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)  # (batch, embed_dim)
        
        # Transform features
        features = self.transformer_layers(pooled)  # (batch, hidden_dim//2)
        
        # Make prediction
        prediction = self.prediction_head(features).squeeze()  # (batch,)
        
        return prediction

def setup_distributed():
    """Initialize the distributed environment."""
    # These will be set by torchrun
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    
    # Initialize the process group
    dist.init_process_group("nccl")
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    return local_rank, world_size, rank

def cleanup_distributed():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_model_torchrun():
    """Train the model using torchrun."""
    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    
    # Set device for this process
    device = torch.device(f'cuda:{local_rank}')
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Torchrun training of BioNeMo committor model with backbone structure tokens')
    parser.add_argument('--data_path', type=str, default='data_processed/path_atlas_tokenized_backbone_full.h5',
                        help='Path to the tokenized backbone dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/bionemo_backbone_committor_torchrun',
                        help='Directory to save checkpoints')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--embed_dim', type=int, default=256, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    parser.add_argument('--log_interval', type=int, default=10, help='How often to log training progress')
    
    args = parser.parse_args()
    
    if rank == 0:
        print(f"Initializing distributed training on {world_size} GPUs")
        print(f"Using device: {device}")
    
    # Create dataset
    if rank == 0:
        print(f'Loading dataset from {args.data_path}')
    
    full_dataset = BackboneTokenDataset(args.data_path, max_samples=args.max_samples)
    
    # Create distributed sampler
    train_sampler = DistributedSampler(full_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Create data loader
    train_loader = DataLoader(
        full_dataset, 
        batch_size=args.batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print(f'Dataset size: {len(full_dataset)}')
        print(f'Batch size per GPU: {args.batch_size}')
        print(f'Total batch size: {args.batch_size * world_size}')
    
    # Create model and move to GPU
    model = BioNeMoBackboneCommittorModel(
        vocab_size=4096,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank])
    
    if rank == 0:
        print(f'Model created and wrapped with DDP')
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Total parameters: {total_params:,}')
        print(f'Trainable parameters: {trainable_params:,}')
    
    # Define loss and optimizer
    criterion = nn.MSELoss()  # Using MSE for continuous committor values
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Create checkpoint directory (only on rank 0)
    if rank == 0:
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        
        # Save arguments
        with open(os.path.join(args.checkpoint_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    
    # Training loop
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.num_epochs):
        # Set epoch for sampler
        train_sampler.set_epoch(epoch)
        
        # Training phase
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            structure_tokens = batch['structure_tokens'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            predictions = model(structure_tokens)
            loss = criterion(predictions, labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
            
            if rank == 0 and batch_idx % args.log_interval == 0:
                print(f'Epoch {epoch+1}/{args.num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Calculate average loss
        avg_train_loss = train_loss / num_batches
        
        # Learning rate scheduling
        scheduler.step(avg_train_loss)
        
        # Save checkpoint (only on rank 0)
        if rank == 0:
            print(f'Epoch {epoch+1}/{args.num_epochs}:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Current LR: {optimizer.param_groups[0]["lr"]:.6f}')
            
            # Save checkpoint
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, checkpoint_path)
            
            # Save best model
            if avg_train_loss < best_loss:
                best_loss = avg_train_loss
                best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pt')
                torch.save(model.module.state_dict(), best_model_path)
                print(f'  Saved new best model with loss: {best_loss:.4f}')
                patience_counter = 0
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= 10:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Clean up
    cleanup_distributed()
    
    if rank == 0:
        print('Training completed successfully!')
        print(f'Checkpoints saved to {args.checkpoint_dir}')
        print(f'Best training loss: {best_loss:.4f}')

if __name__ == '__main__':
    train_model_torchrun()