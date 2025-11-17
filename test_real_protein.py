#!/usr/bin/env python3
"""Test pipeline on real GPCR protein (3P0G - β2-adrenergic receptor)"""
import sys
import torch
import numpy as np
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent / "committor_training"))
sys.path.append(str(Path(__file__).parent / "analysis"))

from train_committor import CommittorModel
from aimmd_reweighting import AIMDReweighting

def extract_backbone_from_cif(cif_path):
    """Extract backbone coordinates from CIF file"""
    print(f"Reading {cif_path}...")
    
    coords = []
    with open(cif_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                parts = line.split()
                if len(parts) > 10:
                    atom_name = parts[3]
                    # Get backbone atoms (N, CA, C)
                    if atom_name in ['N', 'CA', 'C']:
                        try:
                            x = float(parts[10])
                            y = float(parts[11])
                            z = float(parts[12])
                            coords.append([x, y, z])
                        except:
                            continue
    
    coords = np.array(coords)
    print(f"  Extracted {len(coords)} backbone atoms")
    
    # Reshape to (n_residues, 3, 3) for N,CA,C
    if len(coords) % 3 == 0:
        coords = coords.reshape(-1, 3, 3)
        print(f"  Reshaped to {coords.shape[0]} residues")
    
    return coords

def tokenize_structure(coords):
    """Simple tokenization: discretize coordinates"""
    # Flatten and normalize
    flat = coords.reshape(-1, 3)
    
    # Center
    flat = flat - flat.mean(axis=0)
    
    # Discretize to token range [0, 4095]
    min_val = flat.min()
    max_val = flat.max()
    normalized = (flat - min_val) / (max_val - min_val)
    tokens = (normalized.mean(axis=1) * 4095).astype(np.int32)
    
    # Pad/truncate to 256
    if len(tokens) > 256:
        tokens = tokens[:256]
    else:
        tokens = np.pad(tokens, (0, 256 - len(tokens)), constant_values=-1)
    
    return tokens

def test_protein(cif_path, model_path):
    """Test complete pipeline on real protein"""
    print("\n=== TESTING REAL PROTEIN: 3P0G ===")
    print("β2-adrenergic receptor (active state)")
    print("")
    
    # Extract coordinates
    coords = extract_backbone_from_cif(cif_path)
    
    # Tokenize
    print("\nTokenizing structure...")
    tokens = tokenize_structure(coords)
    print(f"  Tokens: {len(tokens)} (range: {tokens[tokens>=0].min()} to {tokens[tokens>=0].max()})")
    
    # Load model
    print("\nLoading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CommittorModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Model loaded on {device}")
    
    # Predict committor
    print("\nPredicting committor...")
    with torch.no_grad():
        tokens_tensor = torch.from_numpy(tokens).long().unsqueeze(0).to(device)
        logits = model(tokens_tensor)
        p_B = torch.sigmoid(logits).item()
    
    print(f"\n=== RESULTS ===")
    print(f"Committor p_B: {p_B:.4f}")
    
    if p_B < 0.2:
        state = "INACTIVE"
    elif p_B > 0.8:
        state = "ACTIVE"
    else:
        state = "TRANSITION"
    
    print(f"Predicted state: {state}")
    print(f"Confidence: {abs(p_B - 0.5) * 2:.1%}")
    
    # Interpretation
    print(f"\nInterpretation:")
    if state == "ACTIVE":
        print("  ✓ Protein is in active conformation")
        print("  ✓ Ready for G-protein binding")
        print("  ✓ High probability of signal transduction")
    elif state == "INACTIVE":
        print("  • Protein is in inactive conformation")
        print("  • Not ready for G-protein binding")
        print("  • Low probability of signal transduction")
    else:
        print("  • Protein is in transition state")
        print("  • Intermediate conformation")
        print("  • Potential allosteric site visible")
    
    return p_B, state

if __name__ == '__main__':
    cif_path = 'data_raw/test_protein/3P0G.cif'
    model_path = 'checkpoints/fixed_committor/best_model.pt'
    
    p_B, state = test_protein(cif_path, model_path)
    
    print(f"\n✓ Test complete!")
    print(f"3P0G committor: {p_B:.4f} ({state})")
