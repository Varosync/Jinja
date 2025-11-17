#!/usr/bin/env python3
"""
Allosteric Site Discovery
Identifies binding pockets using committor-based analysis
"""
import numpy as np
import h5py
from pathlib import Path

class AllostericSiteDiscovery:
    """Discover allosteric sites from committor predictions"""
    
    def __init__(self, reweighting_results_path):
        """Load AIMMD reweighting results"""
        data = np.load(reweighting_results_path)
        self.p_B = data['p_B']
        self.weights = data['weights']
        print(f"✓ Loaded {len(self.p_B)} frames")
    
    def identify_transition_state_frames(self, threshold=0.05):
        """
        Find frames near transition state (p_B ≈ 0.5)
        These frames show the protein in critical conformations
        """
        ts_mask = np.abs(self.p_B - 0.5) < threshold
        ts_indices = np.where(ts_mask)[0]
        print(f"✓ Found {len(ts_indices)} transition state frames")
        return ts_indices
    
    def compute_residue_importance(self, coordinates_h5_path, ts_indices):
        """
        Compute residue importance based on structural changes at TS
        
        Args:
            coordinates_h5_path: Path to 3D coordinates HDF5
            ts_indices: Transition state frame indices
        """
        print("Computing residue importance...")
        
        with h5py.File(coordinates_h5_path, 'r') as f:
            # Get all position datasets
            all_datasets = []
            f.visit(lambda name: all_datasets.append(name) if name.endswith('/positions') else None)
            
            if len(all_datasets) == 0:
                print("⚠ No position data found")
                return None
            
            # Load TS frames
            ts_coords = []
            for idx in ts_indices[:100]:  # Limit to 100 frames
                if idx < len(all_datasets):
                    coords = f[all_datasets[idx]][:]
                    ts_coords.append(coords)
            
            if len(ts_coords) == 0:
                print("⚠ No TS coordinates loaded")
                return None
            
            ts_coords = np.array(ts_coords)
            print(f"  Loaded {len(ts_coords)} TS frames")
            
            # Compute RMSF (root mean square fluctuation) per residue
            mean_coords = ts_coords.mean(axis=0)
            rmsf = np.sqrt(((ts_coords - mean_coords)**2).mean(axis=0).mean(axis=-1))
            
            # High RMSF = important for transition
            importance_scores = rmsf / rmsf.max()
            
            return importance_scores
    
    def identify_allosteric_sites(self, importance_scores, top_n=10):
        """
        Identify top allosteric sites
        
        Args:
            importance_scores: Per-residue importance
            top_n: Number of top sites to return
        """
        if importance_scores is None:
            return []
        
        # Get top residues
        top_indices = np.argsort(importance_scores)[-top_n:][::-1]
        
        sites = []
        for idx in top_indices:
            sites.append({
                'residue_index': int(idx),
                'importance': float(importance_scores[idx])
            })
        
        print(f"\n=== TOP {top_n} ALLOSTERIC SITES ===")
        for i, site in enumerate(sites, 1):
            print(f"{i}. Residue {site['residue_index']}: {site['importance']:.3f}")
        
        return sites
    
    def compute_contact_map(self, coordinates_h5_path, ts_indices, cutoff=8.0):
        """
        Compute contact map for transition state frames
        Identifies residue-residue interactions
        """
        print(f"Computing contact map (cutoff={cutoff}Å)...")
        
        with h5py.File(coordinates_h5_path, 'r') as f:
            all_datasets = []
            f.visit(lambda name: all_datasets.append(name) if name.endswith('/positions') else None)
            
            if len(all_datasets) == 0:
                return None
            
            # Load first TS frame
            idx = ts_indices[0] if len(ts_indices) > 0 else 0
            if idx >= len(all_datasets):
                idx = 0
            
            coords = f[all_datasets[idx]][:]
            
            # Compute CA-CA distances (assuming coords are backbone N,CA,C)
            if coords.shape[1] == 3:  # Backbone atoms
                ca_coords = coords[:, 1, :]  # CA is middle atom
            else:
                ca_coords = coords.reshape(-1, 3)
            
            n_residues = len(ca_coords)
            contact_map = np.zeros((n_residues, n_residues))
            
            for i in range(n_residues):
                for j in range(i+1, n_residues):
                    dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
                    if dist < cutoff:
                        contact_map[i, j] = 1
                        contact_map[j, i] = 1
            
            print(f"  Contacts: {contact_map.sum() / 2:.0f}")
            return contact_map

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Allosteric Site Discovery')
    parser.add_argument('--reweighting', type=str, required=True, help='Reweighting results NPZ')
    parser.add_argument('--coordinates', type=str, required=True, help='3D coordinates HDF5')
    parser.add_argument('--output', type=str, required=True, help='Output NPZ file')
    parser.add_argument('--top_n', type=int, default=10, help='Number of top sites')
    args = parser.parse_args()
    
    discovery = AllostericSiteDiscovery(args.reweighting)
    
    # Find transition state frames
    ts_indices = discovery.identify_transition_state_frames()
    
    # Compute residue importance
    importance = discovery.compute_residue_importance(args.coordinates, ts_indices)
    
    # Identify sites
    sites = discovery.identify_allosteric_sites(importance, args.top_n)
    
    # Compute contact map
    contact_map = discovery.compute_contact_map(args.coordinates, ts_indices)
    
    # Save results
    np.savez(args.output,
            sites=sites,
            importance_scores=importance,
            contact_map=contact_map,
            ts_indices=ts_indices)
    
    print(f"\n✓ Results saved to {args.output}")

if __name__ == '__main__':
    main()
