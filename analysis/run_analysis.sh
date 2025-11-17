#!/bin/bash
# Phase 3: Complete AIMMD Analysis Pipeline

set -e

echo "=== PHASE 3: AIMMD ANALYSIS ==="
echo ""

# Paths
MODEL_CHECKPOINT="${1:-../checkpoints/fixed_committor/best_model.pt}"
TOKENIZED_DATA="${2:-../data_processed/sample_tokenized_backbone.h5}"
COORDINATES_DATA="${3:-../data_processed/sample_tokenized_backbone.h5}"
OUTPUT_DIR="${4:-../results/analysis}"

mkdir -p "$OUTPUT_DIR"

echo "Model: $MODEL_CHECKPOINT"
echo "Data: $TOKENIZED_DATA"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: AIMMD Reweighting
echo "Step 1: AIMMD Reweighting..."
python aimmd_reweighting.py \
  --model "$MODEL_CHECKPOINT" \
  --data "$TOKENIZED_DATA" \
  --output "$OUTPUT_DIR/reweighting_results.npz" \
  --kappa 10.0

echo ""

# Step 2: Allosteric Site Discovery
echo "Step 2: Allosteric Site Discovery..."
python allosteric_site_discovery.py \
  --reweighting "$OUTPUT_DIR/reweighting_results.npz" \
  --coordinates "$COORDINATES_DATA" \
  --output "$OUTPUT_DIR/allosteric_sites.npz" \
  --top_n 10

echo ""
echo "=== ANALYSIS COMPLETE ==="
echo "Results saved to: $OUTPUT_DIR"
