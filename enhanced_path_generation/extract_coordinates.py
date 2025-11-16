#!/usr/bin/env python
"""
Extract backbone coordinates (N, CA, C) from CIF files for use with ESM3 structure encoder
"""

import gzip
import numpy as np
from pathlib import Path
import re


def parse_cif_gz(file_path):
    """
    Parse a gzipped CIF file and extract backbone atom coordinates
    
    Args:
        file_path: Path to the .cif.gz file
        
    Returns:
        dict: Dictionary with keys 'N', 'CA', 'C' containing coordinate arrays
    """
    # Initialize dictionaries to store coordinates
    backbone_atoms = {'N': [], 'CA': [], 'C': []}
    
    # Column indices for the atom_site loop
    col_indices = {}
    
    # Open the gzipped file
    with gzip.open(file_path, 'rt') as f:
        lines = f.readlines()
    
    # Find the atom_site loop
    in_atom_loop = False
    col_names = []
    
    for line in lines:
        # Skip empty lines and comments
        if not line.strip() or line.startswith('#'):
            continue
            
        # Check if we're entering the atom_site loop
        if line.startswith('loop_'):
            in_atom_loop = False
            continue
            
        if '_atom_site.' in line and not in_atom_loop:
            in_atom_loop = True
            col_names = []
            
        # Collect column names
        if in_atom_loop and line.startswith('_atom_site.'):
            col_name = line.strip().split('.')[1]
            col_names.append(col_name)
            
        # Map column names to indices
        if in_atom_loop and not line.startswith('_atom_site.') and line.strip() != '' and not line.startswith('loop_'):
            # We've reached the data rows
            if not col_indices:
                # Map column names to indices
                for i, name in enumerate(col_names):
                    col_indices[name] = i
                    
            # Parse data rows
            if line.startswith('ATOM') or line.startswith('HETATM'):
                parts = line.strip().split()
                
                if len(parts) >= max(col_indices.values()) + 1:
                    # Extract relevant fields
                    atom_name = parts[col_indices['label_atom_id']]  # Atom name
                    x = float(parts[col_indices['Cartn_x']])         # X coordinate
                    y = float(parts[col_indices['Cartn_y']])         # Y coordinate
                    z = float(parts[col_indices['Cartn_z']])         # Z coordinate
                    
                    # Check if this is a backbone atom
                    if atom_name in backbone_atoms:
                        backbone_atoms[atom_name].append([x, y, z])
                        
            # Check if we've exited the loop
            if line.startswith('_') and not line.startswith('_atom_site.'):
                in_atom_loop = False
                col_indices = {}  # Reset for potential future loops
    
    # Convert to numpy arrays
    for atom_type in backbone_atoms:
        if backbone_atoms[atom_type]:
            backbone_atoms[atom_type] = np.array(backbone_atoms[atom_type])
        else:
            backbone_atoms[atom_type] = np.array([]).reshape(0, 3)
    
    return backbone_atoms


def extract_backbone_coordinates(cif_file):
    """
    Extract backbone coordinates from a CIF file
    
    Args:
        cif_file: Path to the CIF file (.cif or .cif.gz)
        
    Returns:
        np.ndarray: Array of shape (L, 3, 3) where L is the number of residues
                   and the second dimension represents N, CA, C atoms
    """
    # Parse the CIF file
    backbone_atoms = parse_cif_gz(cif_file)
    
    # Get the number of residues (assume all backbone atoms have the same count)
    num_residues = len(backbone_atoms['CA'])
    
    # Create the coordinate array
    if num_residues > 0:
        coordinates = np.zeros((num_residues, 3, 3))
        
        # Fill in the coordinates
        coordinates[:, 0, :] = backbone_atoms['N'][:num_residues]   # N atoms
        coordinates[:, 1, :] = backbone_atoms['CA'][:num_residues]  # CA atoms
        coordinates[:, 2, :] = backbone_atoms['C'][:num_residues]   # C atoms
        
        return coordinates
    else:
        # Return empty array with correct shape
        return np.array([]).reshape(0, 3, 3)


def test_extraction():
    """Test the extraction with a sample file"""
    # Find a sample CIF file
    cif_dir = Path("data_raw/inactive_structures")
    cif_files = list(cif_dir.glob("*.cif.gz"))
    
    if cif_files:
        sample_file = cif_files[0]
        print(f"Testing extraction with {sample_file}")
        
        coordinates = extract_backbone_coordinates(sample_file)
        print(f"Extracted coordinates shape: {coordinates.shape}")
        print(f"First few coordinates:")
        if coordinates.shape[0] > 0:
            print(coordinates[:3])
        else:
            print("No coordinates extracted")
    else:
        print("No CIF files found for testing")


if __name__ == "__main__":
    test_extraction()


def extract_backbone_coordinates_simple(cif_file):
    """
    Extract backbone coordinates from a CIF file and return simplified representation
    
    Args:
        cif_file: Path to the CIF file (.cif or .cif.gz)
        
    Returns:
        tuple: (coordinates, sequence) where coordinates is array of shape (L, 3, 3)
               and sequence is the protein sequence string
    """
    try:
        # Extract backbone coordinates
        coordinates = extract_backbone_coordinates(cif_file)
        
        # Extract sequence
        sequence = None
        try:
            sequence = extract_sequence_from_cif(cif_file)
        except:
            pass
            
        return coordinates, sequence
    except Exception as e:
        print(f"Warning: Could not extract coordinates from {cif_file}: {e}")
        return None, None


def extract_sequence_from_cif(cif_file):
    """
    Extract protein sequence from CIF file
    
    Args:
        cif_file: Path to CIF file (can be gzipped)
        
    Returns:
        sequence: Protein sequence string or None if not found
    """
    try:
        # Handle gzipped files
        if str(cif_file).endswith('.gz'):
            with gzip.open(cif_file, 'rt') as f:
                content = f.read()
        else:
            with open(cif_file, 'r') as f:
                content = f.read()
        
        # Look for sequence information using a simpler approach
        # Find the line with _entity_poly.pdbx_seq_one_letter_code_can
        lines = content.split('\n')
        
        for i, line in enumerate(lines):
            if '_entity_poly.pdbx_seq_one_letter_code_can' in line:
                # The sequence is in the next non-empty line(s) within semicolons
                # Look for the start of the sequence block
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith(';'):
                    j += 1
                
                if j < len(lines):
                    # Found start of sequence block
                    j += 1  # Skip the line with ';'
                    sequence_lines = []
                    
                    # Collect lines until we find the end of the block
                    while j < len(lines) and not lines[j].strip().endswith(';'):
                        sequence_lines.append(lines[j].strip())
                        j += 1
                    
                    # If we found the end, include the last line (removing the semicolon)
                    if j < len(lines) and lines[j].strip().endswith(';'):
                        last_line = lines[j].strip().rstrip(';')
                        if last_line:
                            sequence_lines.append(last_line)
                    
                    sequence = ''.join(sequence_lines).replace('\n', '').replace(' ', '')
                    return sequence
        
        return None
    except Exception as e:
        print(f"Warning: Could not extract sequence from {cif_file}: {e}")
        return None