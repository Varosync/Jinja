#!/bin/bash
# Training script for committor model with venv

echo "Starting committor model training..."

# Activate venv
source ../venv/bin/activate

DATA_PATH="${1:-../data_processed/path_atlas_tokenized_backbone_full.h5}"
CHECKPOINT_DIR="${2:-../checkpoints}"
BATCH_SIZE=16
NUM_EPOCHS=150
LEARNING_RATE=0.0001

echo "Data: $DATA_PATH"
echo "Checkpoints: $CHECKPOINT_DIR"

torchrun --nproc_per_node=8 --master_port=29500 \
  train_committor.py \
  --data_path "$DATA_PATH" \
  --checkpoint_dir "$CHECKPOINT_DIR" \
  --batch_size $BATCH_SIZE \
  --num_epochs $NUM_EPOCHS \
  --learning_rate $LEARNING_RATE \
  --log_interval 10

echo "Training complete!"
