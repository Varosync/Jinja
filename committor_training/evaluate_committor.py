#!/usr/bin/env python3
"""Evaluate committor model"""
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
import argparse
from path_atlas_dataset_with_structure_tokens import BackboneTokenDataset
from train_committor import CommittorModel

def evaluate(checkpoint_path, data_path, max_samples=None):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    model = CommittorModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    
    # Use DataParallel for multi-GPU inference
    if torch.cuda.device_count() > 1:
        print(f'Using {torch.cuda.device_count()} GPUs for evaluation')
        model = torch.nn.DataParallel(model)
    
    model.eval()
    
    print('Loading dataset...')
    dataset = BackboneTokenDataset(data_path)
    if max_samples:
        dataset = torch.utils.data.Subset(dataset, range(min(max_samples, len(dataset))))
    loader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f'Evaluating on {len(dataset)} samples...')
    
    predictions, labels = [], []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i % 10 == 0:
                print(f'  Batch {i}/{len(loader)}')
            logits = model(batch['structure_tokens'].to(device))
            preds = torch.sigmoid(logits)
            predictions.extend(preds.cpu().numpy())
            labels.extend(batch['labels'].numpy())
    
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    print('\n=== COMMITTOR MODEL EVALUATION ===')
    print(f'MSE: {mean_squared_error(labels, predictions):.6f}')
    print(f'MAE: {mean_absolute_error(labels, predictions):.4f}')
    print(f'R²: {r2_score(labels, predictions):.6f}')
    print(f'Pearson: {pearsonr(labels, predictions)[0]:.6f}')
    print(f'Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]')
    print(f'Prediction std: {predictions.std():.4f}')
    print(f'Label std: {labels.std():.4f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--max_samples', type=int, default=None, help='Limit samples for quick eval')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data, args.max_samples)
