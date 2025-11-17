# Analysis (Phase 3) - AIMMD Reweighting & Site Discovery

## Overview

Implements the AIMMD algorithm (Jung et al. 2023) to:
1. Compute free energy landscapes from committor predictions
2. Identify allosteric binding sites in GPCRs

## Files

**Core Scripts:**
- `aimmd_reweighting.py` - AIMMD reweighting (Eqs. 12, 15, 17)
- `allosteric_site_discovery.py` - Site identification
- `run_analysis.sh` - Complete pipeline

## Usage

**Run Full Analysis:**
```bash
bash run_analysis.sh \
  <model_checkpoint> \
  <tokenized_data> \
  <coordinates_data> \
  <output_dir>
```

**Example:**
```bash
bash run_analysis.sh \
  ../checkpoints/fixed_committor/best_model.pt \
  ../data_processed/sample_tokenized_backbone.h5 \
  ../data_processed/sample_tokenized_backbone.h5 \
  ../results/analysis
```

## AIMMD Algorithm (Jung et al. 2023)

### Step 1: Reweighting

**Equation 12 - AIMMD Weights:**
```
w(x) = exp(-κ |p_B(x) - 0.5|)
```
- κ = bias strength (default 10.0)
- p_B(x) = committor prediction from trained model

**Equation 15 - Free Energy:**
```
F(p_B) = -kT ln(ρ(p_B))
```
- kT = thermal energy (2.5 kJ/mol)
- ρ(p_B) = weighted density

**Equation 17 - Reaction Rate:**
```
k_AB ∝ ∫ δ(p_B - 0.5) exp(-F(p_B)/kT) dp_B
```

### Step 2: Site Discovery

1. **Identify Transition State Frames**: p_B ≈ 0.5
2. **Compute Residue Importance**: RMSF at transition state
3. **Generate Contact Map**: Residue-residue interactions
4. **Rank Sites**: Top N most important residues

## Output

**reweighting_results.npz:**
- `p_B`: Committor predictions for all frames
- `weights`: AIMMD weights
- `bin_centers`: Free energy bin centers
- `free_energy`: Free energy profile F(p_B)
- `rate_constant`: Reaction rate k_AB

**allosteric_sites.npz:**
- `sites`: Top N allosteric sites with importance scores
- `importance_scores`: Per-residue importance
- `contact_map`: Residue contact matrix
- `ts_indices`: Transition state frame indices

## References

Jung, H.; Bolhuis, P. G.; Covino, R. (2023). 
"Molecular Free Energies, Rates, and Mechanisms from Data-Efficient Path Sampling Simulations"
J. Chem. Theory Comput. 19(24), 9045-9053.
