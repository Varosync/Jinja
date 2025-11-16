#!/usr/bin/env python3
"""
Evaluation script for the BioNeMo backbone committor model.
"""

import torch
import h5py
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import argparse
import os
import json
from torch.utils.data import DataLoader
import sys

# Add parent directory to path to import from train script
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from committor_training.train_bionemo_backbone_committor_torchrun import BioNeMoBackboneCommittorModel, BackboneTokenDataset

def evaluate_model(model, test_loader, device):
    """
    Evaluate the model on test data.
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        device (torch.device): Device to run evaluation on
        
    Returns:
        dict: Evaluation metrics
    """
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
    corr, _ = pearsonr(labels, predictions)
    
    # Calculate accuracy using 0.5 threshold
    predicted_classes = (predictions > 0.5).astype(int)
    true_classes = (labels > 0.5).astype(int)
    accuracy = np.mean(predicted_classes == true_classes)
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'r2': float(r2),
        'pearson_correlation': float(corr),
        'accuracy': float(accuracy),
        'mean_prediction': float(np.mean(predictions)),
        'std_prediction': float(np.std(predictions)),
        'mean_label': float(np.mean(labels)),
        'std_label': float(np.std(labels))
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Evaluate BioNeMo backbone committor model')
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--data_path', type=str, default='data_processed/path_atlas_tokenized_backbone_full.h5',
                        help='Path to the tokenized backbone dataset')
    parser.add_argument('--output_dir', type=str, default='results/bionemo_backbone_committor_evaluation',
                        help='Directory to save evaluation results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for evaluation')
    parser.add_argument('--test_split', type=float, default=0.2, help='Proportion of data to use for testing')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to use')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
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
    
    # Evaluate model
    print('Evaluating model...')
    metrics = evaluate_model(model, test_loader, device)
    
    # Print metrics
    print('\n=== Evaluation Results ===')
    print(f'Mean Squared Error: {metrics["mse"]:.4f}')
    print(f'Mean Absolute Error: {metrics["mae"]:.4f}')
    print(f'R² Score: {metrics["r2"]:.4f}')
    print(f'Pearson Correlation: {metrics["pearson_correlation"]:.4f}')
    print(f'Accuracy (0.5 threshold): {metrics["accuracy"]:.4f}')
    print(f'Mean Prediction: {metrics["mean_prediction"]:.4f} ± {metrics["std_prediction"]:.4f}')
    print(f'Mean Label: {metrics["mean_label"]:.4f} ± {metrics["std_label"]:.4f}')
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nMetrics saved to {metrics_path}')
    
    print(f'\nEvaluation completed successfully!')
    print(f'Results saved to {args.output_dir}')

if __name__ == '__main__':
    main()