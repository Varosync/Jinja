#!/usr/bin/env python3
"""Visualize AIMMD analysis results"""
import numpy as np
import matplotlib.pyplot as plt

def plot_free_energy_landscape(results_path, output_path):
    """Plot free energy profile F(p_B)"""
    data = np.load(results_path)
    bin_centers = data['bin_centers']
    F = data['free_energy']
    
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, F, 'b-', linewidth=2)
    plt.axvline(0.5, color='r', linestyle='--', label='Transition State')
    plt.xlabel('Committor p_B', fontsize=14)
    plt.ylabel('Free Energy (kJ/mol)', fontsize=14)
    plt.title('Free Energy Landscape', fontsize=16)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved {output_path}")
    
    # Print key values
    barrier = F.max()
    ts_idx = np.argmin(np.abs(bin_centers - 0.5))
    ts_energy = F[ts_idx]
    print(f"  Barrier height: {barrier:.2f} kJ/mol")
    print(f"  TS energy: {ts_energy:.2f} kJ/mol")

def plot_committor_distribution(results_path, output_path):
    """Plot committor distribution with weights"""
    data = np.load(results_path)
    p_B = data['p_B']
    weights = data['weights']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Unweighted
    ax1.hist(p_B, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('Committor p_B', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Unweighted Distribution', fontsize=14)
    ax1.axvline(0.5, color='r', linestyle='--', label='TS')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Weighted
    ax2.hist(p_B, bins=50, weights=weights, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Committor p_B', fontsize=12)
    ax2.set_ylabel('Weighted Density', fontsize=12)
    ax2.set_title('AIMMD Weighted Distribution', fontsize=14)
    ax2.axvline(0.5, color='r', linestyle='--', label='TS')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved {output_path}")

def plot_allosteric_sites(sites_path, output_path):
    """Plot allosteric site importance"""
    data = np.load(sites_path, allow_pickle=True)
    sites = data['sites']
    
    if len(sites) == 0:
        print("⚠ No sites to plot")
        return
    
    residues = [s['residue_index'] for s in sites]
    importance = [s['importance'] for s in sites]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(residues)), importance, color='coral', edgecolor='black')
    plt.xticks(range(len(residues)), [f"R{r}" for r in residues], rotation=45)
    plt.xlabel('Residue', fontsize=14)
    plt.ylabel('Importance Score', fontsize=14)
    plt.title('Top Allosteric Sites', fontsize=16)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"✓ Saved {output_path}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize AIMMD results')
    parser.add_argument('--reweighting', type=str, required=True)
    parser.add_argument('--sites', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='.')
    args = parser.parse_args()
    
    print("Generating visualizations...")
    
    plot_free_energy_landscape(
        args.reweighting,
        f"{args.output_dir}/free_energy_landscape.png"
    )
    
    plot_committor_distribution(
        args.reweighting,
        f"{args.output_dir}/committor_distribution.png"
    )
    
    plot_allosteric_sites(
        args.sites,
        f"{args.output_dir}/allosteric_sites.png"
    )
    
    print("\n✓ All visualizations complete!")

if __name__ == '__main__':
    main()
