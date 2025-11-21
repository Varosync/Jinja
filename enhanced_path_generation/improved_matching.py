#!/usr/bin/env python
"""
Improved protein matching functions for finding related protein structures
"""

import numpy as np
from pathlib import Path
import re
from collections import defaultdict


def extract_protein_name(filename):
    """
    Extract protein name from filename, handling various naming conventions
    
    Args:
        filename: Name of the file (with or without extension)
        
    Returns:
        protein_name: Extracted protein name
    """
    # Remove extensions
    name = Path(filename).stem
    if name.endswith('.cif'):
        name = name[:-4]
    
    # Handle common naming patterns
    # Pattern: protein_state_otherinfo
    parts = name.split('_')
    if len(parts) >= 2:
        # Assume first part is protein name
        return parts[0].lower()
    
    # Fallback: return cleaned name
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()


def extract_protein_prefix(filename):
    """
    Extract the prefix of a protein name for grouping similar proteins
    
    Args:
        filename: Name of the file
        
    Returns:
        prefix: Protein prefix (first 3-4 characters)
    """
    # Remove extensions
    name = Path(filename).stem
    if name.endswith('.cif'):
        name = name[:-4]
    
    # Take first 3-4 characters as prefix
    # This helps group proteins like 2RH1, 2R4R, 2R4S together
    prefix = re.sub(r'[^a-zA-Z0-9]', '', name)[:4].lower()
    return prefix


def find_matching_proteins_improved(inactive_dir, active_dir):
    """
    Find matching protein pairs between inactive and active directories using improved matching
    
    Args:
        inactive_dir: Directory with inactive structures
        active_dir: Directory with active structures
        
    Returns:
        pairs: List of (inactive_file, active_file) tuples
        groups: Dictionary mapping protein prefixes to lists of (inactive_file, active_file) tuples
    """
    print("Finding matching protein structures (improved matching)...")
    
    # Get all structure files
    inactive_files = list(inactive_dir.glob("*.cif*"))
    active_files = list(active_dir.glob("*.cif*"))
    
    print(f"  Inactive structures: {len(inactive_files)}")
    print(f"  Active structures: {len(active_files)}")
    
    # Group proteins by prefix
    inactive_by_prefix = defaultdict(list)
    active_by_prefix = defaultdict(list)
    
    for f in inactive_files:
        prefix = extract_protein_prefix(f.name)
        inactive_by_prefix[prefix].append(f)
    
    for f in active_files:
        prefix = extract_protein_prefix(f.name)
        active_by_prefix[prefix].append(f)
    
    # Find matches by prefix
    pairs = []
    groups = defaultdict(list)
    
    # Look for direct prefix matches
    common_prefixes = set(inactive_by_prefix.keys()) & set(active_by_prefix.keys())
    print(f"  Common prefixes: {len(common_prefixes)}")
    
    for prefix in common_prefixes:
        inactive_list = inactive_by_prefix[prefix]
        active_list = active_by_prefix[prefix]
        
        # For each inactive structure, pair with each active structure of the same prefix
        for inactive_file in inactive_list:
            for active_file in active_list:
                pair = (inactive_file, active_file)
                pairs.append(pair)
                groups[prefix].append(pair)
                print(f"  Found match: {inactive_file.name} <-> {active_file.name} (prefix: {prefix})")
    
    # Also look for cross-matching within related prefixes
    # This handles cases where similar proteins might have slightly different prefixes
    print("  Looking for cross-matches...")
    cross_match_count = 0
    for inactive_prefix, inactive_list in inactive_by_prefix.items():
        for active_prefix, active_list in active_by_prefix.items():
            # Check if prefixes are similar (share first 3 characters)
            if inactive_prefix[:3] == active_prefix[:3] and inactive_prefix != active_prefix:
                for inactive_file in inactive_list:
                    for active_file in active_list:
                        pair = (inactive_file, active_file)
                        pairs.append(pair)
                        groups[f"{inactive_prefix}_{active_prefix}"].append(pair)
                        cross_match_count += 1
    
    if cross_match_count > 0:
        print(f"  Found {cross_match_count} cross-matches")
    
    print(f"  Total matching pairs: {len(pairs)}")
    return pairs, groups


def main():
    """Test the improved matching function"""
    # Test with actual data directories
    structures_base = Path("data_raw")
    inactive_dir = structures_base / "inactive_structures"
    active_dir = structures_base / "active_structures"
    
    if not inactive_dir.exists() or not active_dir.exists():
        print("Error: Required data directories not found!")
        return 1
    
    # Find matching protein pairs
    pairs, groups = find_matching_proteins_improved(inactive_dir, active_dir)
    
    print(f"\nFound {len(pairs)} total pairs in {len(groups)} groups:")
    for prefix, group_pairs in groups.items():
        print(f"  Group '{prefix}': {len(group_pairs)} pairs")
        for inactive_file, active_file in group_pairs[:3]:  # Show first 3 pairs
            print(f"    {inactive_file.name} <-> {active_file.name}")
        if len(group_pairs) > 3:
            print(f"    ... and {len(group_pairs) - 3} more pairs")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())