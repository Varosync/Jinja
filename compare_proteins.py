#!/usr/bin/env python3
"""Compare active vs inactive GPCR structures"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from test_real_protein import test_protein

print("="*70)
print("COMPARING ACTIVE vs INACTIVE β2-ADRENERGIC RECEPTOR")
print("="*70)

# Test inactive (2RH1)
print("\n1. INACTIVE STATE (2RH1)")
print("-" * 70)
p_B_inactive, state_inactive = test_protein(
    'data_raw/test_protein/2RH1.cif',
    'checkpoints/fixed_committor/best_model.pt'
)

# Test active (3P0G)
print("\n" + "="*70)
print("\n2. ACTIVE STATE (3P0G)")
print("-" * 70)
p_B_active, state_active = test_protein(
    'data_raw/test_protein/3P0G.cif',
    'checkpoints/fixed_committor/best_model.pt'
)

# Summary
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"2RH1 (Inactive): p_B = {p_B_inactive:.4f} → {state_inactive}")
print(f"3P0G (Active):   p_B = {p_B_active:.4f} → {state_active}")
print(f"Δp_B = {abs(p_B_active - p_B_inactive):.4f}")

if p_B_active > p_B_inactive:
    print("\n✓ Model correctly predicts active > inactive")
else:
    print("\n⚠ Unexpected: inactive > active")
