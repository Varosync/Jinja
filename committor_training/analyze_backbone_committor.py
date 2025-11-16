all #!/usr/bin/env python3
"""
Comprehensive analysis script for the BioNeMo backbone committor model.
Provides detailed insights into model performance, data characteristics, and predictions.
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
import argparse
import os
import json
from collections import defaultdict
import pandas as pd

# Add parent directory to path to import from train script
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from committor_training.train_bionemo_backbone_committor_torchrun import BioNeMoBackboneCommittorModel, BackboneTokenDataset
from torch.utils.data import DataLoader

def analyze_data_characteristics(data_path, output_dir, max_samples=1000):
    """
    Analyze characteristics of the tokenized backbone data.
    
    Args:
        data_path (str): Path to the tokenized backbone dataset
        output_dir (str): Directory to save analysis results
        max_samples (int): Maximum number of samples to analyze
    """
    print("Analyzing data characteristics...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(data_path, 'r') as f:
        # Load a subset of data for analysis
        total_samples = len(f['labels'])
        sample_size = min(max_samples, total_samples)
        
        # For h5py, we need to sort the indices
        indices = np.sort(np.random.choice(total_samples, sample_size, replace=False))
        
        # Load data
        labels = f['labels'][indices]
        structure_tokens = f['structure_tokens'][indices]
        
        # Basic statistics
        print(f"  Total samples: {total_samples}")
        print(f"  Analyzed samples: {sample_size}")
        print(f"  Label range: [{labels.min():.3f}, {labels.max():.3f}]")
        print(f"  Structure tokens shape: {structure_tokens.shape}")
        print(f"  Token value range: [{structure_tokens.min()}, {structure_tokens.max()}]")
        
        # Label distribution
        plt.figure(figsize=(10, 6))
        plt.hist(labels, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Committor Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Committor Values')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'committor_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Token statistics
        # Count padding tokens (-1 values)
        padding_count = np.sum(structure_tokens == -1)
        total_tokens = structure_tokens.size
        padding_percentage = (padding_count / total_tokens) * 100
        
        print(f"  Padding tokens: {padding_count} ({padding_percentage:.2f}%)")
        
        # Distribution of non-padding token values
        non_padding_tokens = structure_tokens[structure_tokens != -1]
        plt.figure(figsize=(10, 6))
        plt.hist(non_padding_tokens, bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
        plt.xlabel('Token Value')
        plt.ylabel('Frequency')
        plt.title('Distribution of Non-Padding Structure Token Values')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'token_value_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Sequence length analysis (non-padding tokens per sequence)
        seq_lengths = []
        for i in range(len(structure_tokens)):
            non_padding_count = np.sum(structure_tokens[i] != -1)
            seq_lengths.append(non_padding_count)
        
        plt.figure(figsize=(10, 6))
        plt.hist(seq_lengths, bins=50, alpha=0.7, color='mediumseagreen', edgecolor='black')
        plt.xlabel('Sequence Length (Non-Padding Tokens)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Sequence Lengths')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'sequence_length_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Average sequence length: {np.mean(seq_lengths):.2f}")
        print(f"  Min sequence length: {np.min(seq_lengths)}")
        print(f"  Max sequence length: {np.max(seq_lengths)}")

def analyze_model_predictions(model, test_loader, device, output_dir):
    """
    Analyze model predictions in detail.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        output_dir (str): Directory to save analysis results
    """
    print("Analyzing model predictions...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    all_protein_names = []
    
    with torch.no_grad():
        for batch in test_loader:
            structure_tokens = batch['structure_tokens'].to(device)
            labels = batch['labels'].to(device)
            
            predictions = model(structure_tokens)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    # Prediction error analysis
    errors = predictions - labels
    
    # Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, alpha=0.7, color='gold', edgecolor='black')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'prediction_error_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error vs true value
    plt.figure(figsize=(10, 6))
    plt.scatter(labels, errors, alpha=0.5, color='purple')
    plt.xlabel('True Committor Value')
    plt.ylabel('Prediction Error')
    plt.title('Prediction Error vs True Committor Value')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'error_vs_true.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Binned analysis
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    mean_errors = []
    std_errors = []
    
    for i in range(len(bins) - 1):
        mask = (labels >= bins[i]) & (labels < bins[i + 1])
        if np.sum(mask) > 0:
            bin_errors = errors[mask]
            mean_errors.append(np.mean(bin_errors))
            std_errors.append(np.std(bin_errors))
        else:
            mean_errors.append(0)
            std_errors.append(0)
    
    plt.figure(figsize=(10, 6))
    plt.errorbar(bin_centers, mean_errors, yerr=std_errors, fmt='o-', capsize=5, color='navy')
    plt.xlabel('True Committor Value (Binned)')
    plt.ylabel('Mean Prediction Error')
    plt.title('Mean Prediction Error Across Committor Value Range')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'binned_errors.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Performance metrics by region
    low_mask = labels < 0.3
    mid_mask = (labels >= 0.3) & (labels <= 0.7)
    high_mask = labels > 0.7
    
    regions = ['Low (0.0-0.3)', 'Mid (0.3-0.7)', 'High (0.7-1.0)']
    masks = [low_mask, mid_mask, high_mask]
    
    region_metrics = {}
    for region, mask in zip(regions, masks):
        if np.sum(mask) > 0:
            region_preds = predictions[mask]
            region_labels = labels[mask]
            mse = mean_squared_error(region_labels, region_preds)
            mae = mean_absolute_error(region_labels, region_preds)
            r2 = r2_score(region_labels, region_preds)
            corr, _ = pearsonr(region_labels, region_preds)
            
            region_metrics[region] = {
                'MSE': float(mse),
                'MAE': float(mae),
                'R2': float(r2),
                'Pearson': float(corr),
                'Count': int(np.sum(mask))
            }
    
    # Save region metrics
    with open(os.path.join(output_dir, 'region_metrics.json'), 'w') as f:
        json.dump(region_metrics, f, indent=2)
    
    print("  Region-specific metrics saved to region_metrics.json")

def analyze_feature_importance(model, test_loader, device, output_dir, num_samples=100):
    """
    Analyze which parts of the structure tokens are most important for predictions.
    This is a simplified analysis using gradient-based attribution.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        output_dir (str): Directory to save analysis results
        num_samples (int): Number of samples to analyze
    """
    print("Analyzing feature importance...")
    
    model.eval()
    
    # Get a small batch for analysis
    for batch in test_loader:
        structure_tokens = batch['structure_tokens'][:num_samples].to(device)
        break
    
    # Enable gradients for input
    structure_tokens.requires_grad_(True)
    
    # Forward pass
    predictions = model(structure_tokens)
    
    # Compute gradients with respect to a target (mean prediction)
    target = predictions.mean()
    target.backward()
    
    # Get gradients
    gradients = structure_tokens.grad.abs().mean(dim=0).cpu().numpy()
    
    # Plot gradient importance
    plt.figure(figsize=(12, 6))
    plt.plot(gradients, color='crimson')
    plt.xlabel('Token Position')
    plt.ylabel('Mean Absolute Gradient')
    plt.title('Feature Importance by Token Position')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Feature importance analysis completed for {num_samples} samples")

def generate_committor_trajectory_visualization(data_path, output_dir, num_trajectories=5):
    """
    Generate visualization of committor values along trajectories.
    
    Args:
        data_path (str): Path to the tokenized backbone dataset
        output_dir (str): Directory to save analysis results
        num_trajectories (int): Number of trajectories to visualize
    """
    print("Generating committor trajectory visualization...")
    
    with h5py.File(data_path, 'r') as f:
        # Group samples by protein name to identify trajectories
        protein_names = []
        labels = []
        
        # Load all data (or a subset for efficiency)
        max_samples = min(5000, len(f['labels']))  # Limit for efficiency
        indices = np.sort(np.random.choice(len(f['labels']), max_samples, replace=False))
        
        for i in indices:
            protein_name = f['protein_names'][i]
            label = f['labels'][i]
            
            protein_names.append(protein_name.decode('utf-8') if isinstance(protein_name, bytes) else str(protein_name))
            labels.append(label)
        
        # Group by protein name
        protein_groups = defaultdict(list)
        for name, label in zip(protein_names, labels):
            protein_groups[name].append(label)
        
        # Select trajectories for visualization
        selected_proteins = list(protein_groups.keys())[:num_trajectories]
        
        plt.figure(figsize=(12, 8))
        for i, protein_name in enumerate(selected_proteins):
            committor_values = sorted(protein_groups[protein_name])
            positions = np.linspace(0, 1, len(committor_values))
            
            plt.plot(positions, committor_values, marker='o', markersize=4, label=f'{protein_name}', alpha=0.7)
        
        plt.xlabel('Trajectory Progress')
        plt.ylabel('Committor Value')
        plt.title('Committor Values Along Protein Trajectories')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'trajectory_committor_values.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"  Trajectory visualization completed for {num_trajectories} proteins")

def main():
    parser = argparse.ArgumentParser(description='Comprehensive analysis of BioNeMo backbone committor model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, default='data_processed/path_atlas_tokenized_backbone_full.h5',
                        help='Path to the tokenized backbone dataset')
    parser.add_argument('--output_dir', type=str, default='analysis/bionemo_backbone_committor',
                        help='Directory to save analysis results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=1000, help='Maximum number of samples for data analysis')
    parser.add_argument('--test_split', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--analyze_predictions', action='store_true', 
                        help='Analyze model predictions in detail')
    parser.add_argument('--analyze_features', action='store_true',
                        help='Analyze feature importance')
    parser.add_argument('--analyze_trajectories', action='store_true',
                        help='Generate trajectory visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Analyze data characteristics
    analyze_data_characteristics(args.data_path, args.output_dir, args.max_samples)
    
    if args.analyze_predictions or args.analyze_features or args.analyze_trajectories:
        # Load model
        print(f'Loading model from {args.checkpoint_path}')
        model = BioNeMoBackboneCommittorModel()
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
        model.to(device)
        model.eval()
        print('Model loaded successfully')
        
        # Create dataset
        print(f'Loading dataset from {args.data_path}')
        full_dataset = BackboneTokenDataset(args.data_path, max_samples=args.max_samples)
        print(f'Full dataset size: {len(full_dataset)}')
        
        # Split into test set
        test_size = int(len(full_dataset) * args.test_split)
        _, test_dataset = torch.utils.data.random_split(
            full_dataset, 
            [len(full_dataset) - test_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        print(f'Test set size: {len(test_dataset)}')
        
        # Create data loader
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
        if args.analyze_predictions:
            # Analyze model predictions
            analyze_model_predictions(model, test_loader, device, args.output_dir)
        
        if args.analyze_features:
            # Analyze feature importance
            analyze_feature_importance(model, test_loader, device, args.output_dir)
        
        if args.analyze_trajectories:
            # Generate trajectory visualizations
            generate_committor_trajectory_visualization(args.data_path, args.output_dir)
    
    print(f'\nAnalysis completed successfully!')
    print(f'Results saved to {args.output_dir}')

if __name__ == '__main__':
    main()