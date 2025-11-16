#!/bin/bash

# Script to run distributed tokenization using all 8 GPUs
# This script splits the work across multiple processes

# Activate conda environment
source /home/nebius/miniconda3/bin/activate bionemo_skelda

# Set number of processes (GPUs)
NUM_PROCESSES=8

echo "Starting distributed tokenization with $NUM_PROCESSES processes"

# Run distributed tokenization
torchrun --nproc_per_node=$NUM_PROCESSES \
    enhanced_path_generation/tokenize_atlas_backbone_simple_distributed.py \
    --input data_processed/path_atlas_backbone.h5 \
    --output data_processed/path_atlas_tokenized_backbone_full.h5

echo "Tokenization completed"