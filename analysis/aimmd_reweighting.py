#!/usr/bin/env python3
"""
AIMMD Reweighting Algorithm (Jung et al. 2023)
Computes free energy landscapes from committor predictions
"""
import torch
import numpy as np
import h5py
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent / "committor_training"))
from train_committor import CommittorModel

class AIMDReweighting:
    """Implements AIMMD reweighting from Jung et al. 2023, Eqs. 12, 15, 17"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = CommittorModel().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Loaded model on {self.device}")
    
    def predict_committor(self, structure_tokens):
        """Predict p_B for structure tokens"""
        with torch.no_grad():
            tokens = torch.from_numpy(structure_tokens).long().to(self.device)
            if tokens.dim() == 1:
                tokens = tokens.unsqueeze(0)
            logits = self.model(tokens)
            p_B = torch.sigmoid(logits)
        return p_B.cpu().numpy()
    
    def compute_weights(self, p_B_values, kappa=10.0):
        """
        Compute AIMMD weights (Jung et al. 2023, Eq. 12)
        w(x) = exp(-kappa * |p_B(x) - 0.5|)
        
        Args:
            p_B_values: Committor predictions [0,1]
            kappa: Bias strength (default 10.0)
        """
        weights = np.exp(-kappa * np.abs(p_B_values - 0.5))
        return weights / weights.sum()  # Normalize
    
    def compute_free_energy(self, p_B_values, weights, n_bins=50, kT=2.5):
        """
        Compute free energy profile F(p_B) (Jung et al. 2023, Eq. 15)
        F(p_B) = -kT * ln(ρ(p_B))
        
        Args:
            p_B_values: Committor predictions
            weights: AIMMD weights
            n_bins: Number of bins for histogram
            kT: Thermal energy (kJ/mol)
        """
        hist, bin_edges = np.histogram(p_B_values, bins=n_bins, weights=weights, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Avoid log(0)
        hist = np.maximum(hist, 1e-10)
        
        # Free energy
        F = -kT * np.log(hist)
        F = F - F.min()  # Set minimum to 0
        
        return bin_centers, F
    
    def compute_reaction_rate(self, p_B_values, weights, kT=2.5):
        """
        Compute reaction rate constant (Jung et al. 2023, Eq. 17)
        k_AB ∝ ∫ δ(p_B - 0.5) exp(-F(p_B)/kT) dp_B
        """
        # Find transition state (p_B ≈ 0.5)
        ts_mask = np.abs(p_B_values - 0.5) < 0.05
        if ts_mask.sum() == 0:
            return 0.0
        
        # Rate proportional to weighted density at TS
        k_AB = weights[ts_mask].sum()
        return k_AB
    
    def process_trajectory(self, tokenized_h5_path, output_path):
        """Process full trajectory and compute free energy landscape"""
        print(f"Processing {tokenized_h5_path}...")
        
        with h5py.File(tokenized_h5_path, 'r') as f:
            structure_tokens = f['structure_tokens'][:]
            n_frames = len(structure_tokens)
            print(f"  Frames: {n_frames}")
            
            # Predict committor for all frames
            print("  Computing committor predictions...")
            p_B_values = []
            batch_size = 100
            for i in range(0, n_frames, batch_size):
                batch = structure_tokens[i:i+batch_size]
                p_B_batch = self.predict_committor(batch)
                p_B_values.extend(p_B_batch)
            
            p_B_values = np.array(p_B_values)
            print(f"  p_B range: [{p_B_values.min():.3f}, {p_B_values.max():.3f}]")
            
            # Compute AIMMD weights
            print("  Computing AIMMD weights...")
            weights = self.compute_weights(p_B_values)
            
            # Compute free energy
            print("  Computing free energy landscape...")
            bin_centers, F = self.compute_free_energy(p_B_values, weights)
            
            # Compute rate
            k_AB = self.compute_reaction_rate(p_B_values, weights)
            print(f"  Reaction rate constant: {k_AB:.6f}")
            
            # Save results
            print(f"  Saving to {output_path}...")
            np.savez(output_path,
                    p_B=p_B_values,
                    weights=weights,
                    bin_centers=bin_centers,
                    free_energy=F,
                    rate_constant=k_AB)
            
            print("✓ Complete!")
            return {
                'p_B': p_B_values,
                'weights': weights,
                'free_energy': (bin_centers, F),
                'rate': k_AB
            }

def main():
    import argparse
    parser = argparse.ArgumentParser(description='AIMMD Reweighting')
    parser.add_argument('--model', type=str, required=True, help='Trained model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Tokenized HDF5 file')
    parser.add_argument('--output', type=str, required=True, help='Output NPZ file')
    parser.add_argument('--kappa', type=float, default=10.0, help='Bias strength')
    args = parser.parse_args()
    
    reweighter = AIMDReweighting(args.model)
    results = reweighter.process_trajectory(args.data, args.output)
    
    print(f"\n=== RESULTS ===")
    print(f"Free energy barrier: {results['free_energy'][1].max():.2f} kJ/mol")
    print(f"Rate constant: {results['rate']:.6f}")

if __name__ == '__main__':
    main()
