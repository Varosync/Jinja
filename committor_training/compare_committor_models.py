#!/usr/bin/env python3
"""
Script to compare performance of different committor models.
Compares backbone structure token model with other approaches.
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import argparse
import os
import json
from collections import defaultdict

# Add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our backbone model
from committor_training.train_bionemo_backbone_committor import BioNeMoBackboneCommittorModel, BackboneTokenDataset
from torch.utils.data import DataLoader

# For comparison, we'll create a simple baseline model
import torch.nn as nn

class BaselineCommittorModel(nn.Module):
    """Simple baseline model for comparison."""
    
    def __init__(self, vocab_size=4096, embed_dim=128, hidden_dim=256):
        super(BaselineCommittorModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=4095)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, structure_tokens):
        # structure_tokens shape: (batch, seq_len)
        embedded = self.embedding(structure_tokens)  # (batch, seq_len, embed_dim)
        # Average pooling over sequence dimension
        pooled = torch.mean(embedded, dim=1)  # (batch, embed_dim)
        out = self.linear1(pooled)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.sigmoid(out)
        return out.squeeze()  # (batch,)

def evaluate_model(model, test_loader, device, model_name):
    """
    Evaluate a model and return metrics.
    
    Args:
        model (nn.Module): Model to evaluate
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        model_name (str): Name of the model for reporting
        
    Returns:
        dict: Evaluation metrics
    """
    print(f"Evaluating {model_name}...")
    
    model.eval()
    all_predictions = []
    all_labels = []
    
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
    
    # Calculate metrics
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    r2 = r2_score(labels, predictions)
    pearson_corr, _ = pearsonr(labels, predictions)
    
    metrics = {
        'model_name': model_name,
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_correlation': float(pearson_corr)
    }
    
    return metrics

def plot_comparison(metrics_list, output_dir):
    """
    Plot comparison of model performances.
    
    Args:
        metrics_list (list): List of metric dictionaries
        output_dir (str): Directory to save plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract model names and metrics
    model_names = [m['model_name'] for m in metrics_list]
    mse_values = [m['mse'] for m in metrics_list]
    mae_values = [m['mae'] for m in metrics_list]
    r2_values = [m['r2'] for m in metrics_list]
    pearson_values = [m['pearson_correlation'] for m in metrics_list]
    
    # Plot MSE comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, mse_values, color=['skyblue', 'lightcoral'])
    plt.ylabel('Mean Squared Error')
    plt.title('Model Comparison - MSE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mse_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_mse.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot MAE comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, mae_values, color=['lightgreen', 'orange'])
    plt.ylabel('Mean Absolute Error')
    plt.title('Model Comparison - MAE')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, mae_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_mae.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot correlation comparison
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, pearson_values, color=['mediumpurple', 'gold'])
    plt.ylabel('Pearson Correlation')
    plt.title('Model Comparison - Pearson Correlation')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, pearson_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'comparison_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save metrics to CSV for detailed analysis
    import pandas as pd
    df = pd.DataFrame(metrics_list)
    df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)

def main():
    parser = argparse.ArgumentParser(description='Compare performance of different committor models')
    parser.add_argument('--backbone_checkpoint', type=str, required=True,
                        help='Path to the backbone model checkpoint')
    parser.add_argument('--data_path', type=str, default='data_processed/path_atlas_tokenized_backbone_full.h5',
                        help='Path to the tokenized backbone dataset')
    parser.add_argument('--output_dir', type=str, default='comparison/committor_models',
                        help='Directory to save comparison results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=2000, help='Maximum number of samples to test')
    parser.add_argument('--test_split', type=float, default=0.3, help='Proportion of data to use for testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
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
    
    # Evaluate backbone model
    print(f'Loading backbone model from {args.backbone_checkpoint}')
    backbone_model = BioNeMoBackboneCommittorModel()
    backbone_model.load_state_dict(torch.load(args.backbone_checkpoint, map_location=device))
    backbone_model.to(device)
    backbone_model.eval()
    
    backbone_metrics = evaluate_model(backbone_model, test_loader, device, "Backbone Structure Tokens")
    
    # Evaluate baseline model
    print('Creating and evaluating baseline model...')
    baseline_model = BaselineCommittorModel()
    baseline_model.to(device)
    baseline_model.eval()
    
    baseline_metrics = evaluate_model(baseline_model, test_loader, device, "Baseline (Simple Embedding)")
    
    # Compare models
    metrics_list = [backbone_metrics, baseline_metrics]
    
    # Print comparison
    print("\nModel Performance Comparison:")
    print("=" * 50)
    for metrics in metrics_list:
        print(f"\n{metrics['model_name']}:")
        print(f"  MSE: {metrics['mse']:.6f}")
        print(f"  MAE: {metrics['mae']:.6f}")
        print(f"  R²:  {metrics['r2']:.6f}")
        print(f"  Pearson Correlation: {metrics['pearson_correlation']:.6f}")
    
    # Plot comparison
    plot_comparison(metrics_list, args.output_dir)
    
    # Save detailed metrics
    comparison_path = os.path.join(args.output_dir, 'detailed_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(metrics_list, f, indent=2)
    
    print(f'\nComparison completed successfully!')
    print(f'Results saved to {args.output_dir}')
    print(f'Detailed metrics saved to {comparison_path}')

if __name__ == '__main__':
    main()