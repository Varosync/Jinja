#!/usr/bin/env python3
"""
Dataset class for handling tokenized protein structures with structure tokens.
This dataset is designed to work with HDF5 files containing structure tokens
generated from the ESM3 dVAE encoder.
"""

import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class BackboneTokenDataset(Dataset):
    """Dataset for backbone structure tokens."""
    
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


# For backward compatibility, also define the old class name
PathAtlasDatasetWithStructureTokens = BackboneTokenDataset


if __name__ == "__main__":
    # Example usage
    dataset = BackboneTokenDataset('data_processed/path_atlas_tokenized_backbone_full.h5')
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Structure tokens shape: {sample['structure_tokens'].shape}")
        print(f"Label: {sample['labels']}")
        print(f"Sequence: {sample['sequences'][:50]}...")
        print(f"Protein name: {sample['protein_names']}")