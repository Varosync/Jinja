# BioNeMo Committor Training with Backbone Structure Tokens

This directory contains scripts for training and evaluating committor models using backbone structure tokens derived from the ESM3 structure encoder.

## Overview

We've implemented an improved pipeline for training committor models that:
1. Uses all available training data (not just a small subset)
2. Employs better hyperparameters for training stability
3. Includes comprehensive logging and monitoring
4. Provides detailed evaluation metrics

## Key Components

### 1. Improved Training Script
- Script: `train_bionemo_backbone_committor_improved.py`
- Uses all available samples in the dataset
- Implements proper logging to track training progress
- Uses a lower learning rate (0.0001) for better stability
- Saves training logs for analysis

### 2. Training Runner
- Script: `run_improved_training.sh`
- Configures and runs distributed training with torchrun
- Easily adjustable parameters for different training configurations

### 3. Evaluation Scripts
- Script: `evaluate_bionemo_backbone_committor_improved.py`
- Comprehensive evaluation with multiple metrics
- Generates visualizations of predictions vs. true values
- Calculates region-specific metrics for detailed analysis

### 4. Evaluation Runner
- Script: `run_evaluation_improved.sh`
- Configures and runs the evaluation pipeline

## Usage

### Training

To train the improved model with distributed training:

```bash
./committor_training/run_improved_training.sh
```

This will:
- Use all samples in the dataset (not just a subset)
- Train for 100 epochs with early stopping
- Use a learning rate of 0.0001
- Save checkpoints and training logs

### Evaluation

To evaluate the trained model:

```bash
./committor_training/run_evaluation_improved.sh
```

This will:
- Evaluate the model on the full dataset
- Generate performance metrics (MSE, RMSE, R²)
- Create visualizations of predictions vs. true values
- Calculate region-specific metrics
- Save all results to the analysis directory

## Expected Improvements

The improved training approach should address the issues observed with the previous model:

1. **Better Data Utilization**: Using all 31,850 samples instead of just 1,000
2. **Stable Training**: Lower learning rate and proper gradient clipping
3. **Comprehensive Monitoring**: Training logs to track convergence
4. **Detailed Evaluation**: Region-specific metrics to understand model performance across different committor value ranges

## Model Architecture

The improved model uses the same architecture as the previous version:
1. Embedding layer for structure tokens (4096 vocab size)
2. Positional encoding
3. Transformer-like layers for feature extraction
4. Attention mechanism to focus on important regions
5. Prediction head with sigmoid activation for continuous committor values (0-1)

## Results Directory

Results are saved to `analysis/bionemo_backbone_committor_improved/`:
- `metrics.json`: Overall performance metrics
- `region_metrics.json`: Performance metrics for different committor value ranges
- `predictions_vs_labels.png`: Scatter plot of predictions vs. true values
- `distributions.png`: Distribution plots of predictions, labels, and errors
- `predictions.json`: Raw predictions and labels for further analysis
- `training_log.txt`: Training progress logs (when using the improved training script)