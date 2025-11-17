# Committor Training (Phase 2) - COMPLETE ✓

## Status: Production Ready (R²=99.55%)

### Files

**Core Scripts:**
- `train_committor.py` - Production training script (3M params, Focal Loss, Transformer)
- `evaluate_committor.py` - Model evaluation
- `path_atlas_dataset_with_structure_tokens.py` - Dataset loader
- `run_training.sh` - Training launcher

### Model Performance

- **MSE**: 0.000377
- **MAE**: 0.0157
- **R²**: 0.9955 (99.55%)
- **Pearson**: 0.9980
- **Prediction Range**: [0.06, 0.96] (full spectrum)
- **No Mode Collapse** ✓

### Usage

**Train:**
```bash
bash run_training.sh <data_path> <checkpoint_dir>
```

**Evaluate:**
```bash
python evaluate_committor.py \
  --checkpoint ../checkpoints/fixed_committor/best_model.pt \
  --data ../data_processed/sample_tokenized_backbone.h5
```

### Model Architecture

```
CommittorModel (3,056,898 params)
├── Embedding (4096 vocab → 256d)
├── Positional Encoding
├── Transformer Encoder (3 layers, 8 heads)
├── Mean Pooling
├── MLP Head (512→256→128→1)
└── Temperature Scaling (T=1.5)
```

### Key Improvements Over Baseline

1. **Focal Loss** - Fixed mode collapse
2. **Transformer Encoder** - Better features
3. **Temperature Scaling** - Calibrated predictions
4. **OneCycleLR** - Optimal learning rate
5. **Higher Dropout (0.3)** - Prevents overfitting

### Next: Phase 3 - Analysis

Move to `../analysis/` for:
- AIMMD reweighting algorithm
- Free energy landscape computation
- Allosteric site discovery
