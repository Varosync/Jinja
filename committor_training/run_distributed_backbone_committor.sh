#!/bin/bash

# Distributed training script for BioNeMo backbone committor model
# Uses all 8 H100 GPUs for maximum throughput

echo "Starting distributed training of BioNeMo backbone committor model..."
echo "Using all 8 H100 GPUs"

# Set environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export PYTHONPATH=/home/nebius/skelda:$PYTHONPATH

# Training parameters
DATA_PATH="data_processed/path_atlas_tokenized_backbone_full.h5"
CHECKPOINT_DIR="checkpoints/bionemo_backbone_committor_distributed"
BATCH_SIZE_PER_GPU=32
LEARNING_RATE=0.001
NUM_EPOCHS=100
EMBED_DIM=256
HIDDEN_DIM=512
DROPOUT=0.1

# Calculate total batch size (batch_size_per_gpu * num_gpus)
TOTAL_BATCH_SIZE=$((BATCH_SIZE_PER_GPU * 8))
echo "Total batch size: $TOTAL_BATCH_SIZE (32 per GPU * 8 GPUs)"

# Create checkpoint directory
mkdir -p $CHECKPOINT_DIR

# Run distributed training with torchrun
torchrun --nproc_per_node=8 \
  --nnodes=1 \
  --node_rank=0 \
  --master_addr="localhost" \
  --master_port=12345 \
  committor_training/train_bionemo_backbone_committor_distributed.py \
  --data_path $DATA_PATH \
  --checkpoint_dir $CHECKPOINT_DIR \
  --batch_size $BATCH_SIZE_PER_GPU \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --embed_dim $EMBED_DIM \
  --hidden_dim $HIDDEN_DIM \
  --dropout $DROPOUT \
  --validation_split 0.1 \
  --log_interval 10

echo "Distributed training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"