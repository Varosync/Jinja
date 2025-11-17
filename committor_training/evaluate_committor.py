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

def evaluate(checkpoint_path, data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = CommittorModel().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    dataset = BackboneTokenDataset(data_path)
    loader = DataLoader(dataset, batch_size=100, shuffle=False)
    
    predictions, labels = [], []
    with torch.no_grad():
        for batch in loader:
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
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data)
