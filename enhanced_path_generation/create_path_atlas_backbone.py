#!/usr/bin/env python
"""
Create path atlas HDF5 file from generated NPZ files with backbone coordinates
Combines transition paths and well samples into a single HDF5 file with proper backbone format
"""

import h5py
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import re


def extract_protein_name(filename):
    """
    Extract protein name from filename
    
    Args:
        filename: Name of the file (with or without extension)
        
    Returns:
        protein_name: Extracted protein name
    """
    # Remove extensions
    name = Path(filename).stem
    if name.endswith('.npz'):
        name = name[:-4]
    
    # Handle common naming patterns
    # Pattern: path_protein1_to_protein2 or well_state_protein_repXXX_gpuY
    if name.startswith('path_'):
        # Extract first protein from path_XXX_to_YYY
        parts = name.split('_')[1:]  # Remove 'path'
        if 'to' in parts:
            to_idx = parts.index('to')
            return parts[0].lower()  # Return first protein
    elif name.startswith('well_'):
        # Extract protein from well_state_protein_repXXX_gpuY
        parts = name.split('_')
        if len(parts) >= 3:
            return parts[2].lower()  # Return protein name
    
    # Fallback: return cleaned name
    return re.sub(r'[^a-zA-Z0-9]', '', name).lower()


def load_npz_file(filepath):
    """
    Load data from NPZ file
    
    Args:
        filepath: Path to NPZ file
        
    Returns:
        dict: Data loaded from NPZ file
    """
    data = np.load(filepath, allow_pickle=True)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


def create_path_atlas_backbone(paths_dir, wells_dir, barriers_dir, output_file):
    """
    Create path atlas HDF5 file from transition paths, well samples, and barrier samples
    with proper backbone coordinates format for ESM3
    
    Args:
        paths_dir: Directory containing transition path NPZ files
        wells_dir: Directory containing well sample NPZ files
        barriers_dir: Directory containing barrier sample NPZ files
        output_file: Output HDF5 file path
    """
    print("=" * 60)
    print("CREATING PATH ATLAS HDF5 FILE (BACKBONE COORDINATES)")
    print("=" * 60)
    
    # Find all NPZ files
    path_files = list(Path(paths_dir).glob("*.npz"))
    well_files = list(Path(wells_dir).glob("*.npz"))
    barrier_files = list(Path(barriers_dir).glob("*.npz"))
    
    print(f"Found {len(path_files)} transition path files")
    print(f"Found {len(well_files)} well sample files")
    print(f"Found {len(barrier_files)} barrier sample files")
    print(f"Total files: {len(path_files) + len(well_files) + len(barrier_files)}")
    
    if not path_files and not well_files:
        print("No files found!")
        return False
    
    # Create output file
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(output_file, 'w') as hdf:
        frame_count = 0
        # Keep track of frame IDs to avoid duplicates
        used_frame_ids = set()
        
        # Process transition paths
        print("\nProcessing transition paths...")
        for path_file in tqdm(path_files, desc="Paths"):
            try:
                # Load data
                data = load_npz_file(path_file)
                
                # Extract protein name
                protein_name = extract_protein_name(path_file.name)
                
                # For transition paths, we'll use a committor value between 0 and 1
                # representing progress along the path
                if 'cv_values' in data:
                    cv_values = data['cv_values']
                else:
                    # Fallback to linear progression
                    n_frames = len(data['positions'])
                    cv_values = np.linspace(0.0, 1.0, n_frames)
                
                # Extract sequences if available
                start_sequence = data.get('start_sequence', None)
                end_sequence = data.get('end_sequence', None)
                
                # Process each frame
                for i in range(len(cv_values)):
                    # Generate unique frame ID
                    frame_id = f"transition_{protein_name}_{frame_count:06d}"
                    while frame_id in used_frame_ids:
                        frame_count += 1
                        frame_id = f"transition_{protein_name}_{frame_count:06d}"
                    used_frame_ids.add(frame_id)
                    frame_group = hdf.create_group(frame_id)
                    
                    # Store frame data - positions should already be in (L, 3, 3) format
                    # Check if positions have the correct shape
                    position_data = data['positions'][i]
                    if position_data.ndim != 3 or position_data.shape[1] != 3 or position_data.shape[2] != 3:
                        print(f"  Warning: Skipping frame with incorrect coordinate shape {position_data.shape}")
                        continue
                    frame_group.create_dataset('positions', data=position_data)
                    frame_group.create_dataset('label', data=float(cv_values[i]))  # Committor value
                    
                    # Store sequence information
                    if start_sequence is not None:
                        if isinstance(start_sequence, np.ndarray):
                            # If it's a numpy array, convert to string first
                            frame_group.create_dataset('sequence', data=str(start_sequence).encode('utf-8'))
                        else:
                            frame_group.create_dataset('sequence', data=str(start_sequence).encode('utf-8'))
                    elif end_sequence is not None:
                        if isinstance(end_sequence, np.ndarray):
                            # If it's a numpy array, convert to string first
                            frame_group.create_dataset('sequence', data=str(end_sequence).encode('utf-8'))
                        else:
                            frame_group.create_dataset('sequence', data=str(end_sequence).encode('utf-8'))
                    
                    # Store metadata
                    frame_group.attrs['source'] = 'transition_path'
                    frame_group.attrs['protein_name'] = protein_name
                    frame_group.attrs['frame_index'] = i
                    
                    frame_count += 1
                    
            except Exception as e:
                print(f"Error processing {path_file}: {e}")
                continue
        
        # Process well samples
        print("\nProcessing well samples...")
        for well_file in tqdm(well_files, desc="Wells"):
            try:
                # Load data
                data = load_npz_file(well_file)
                
                # Extract protein name and state type
                protein_name = extract_protein_name(well_file.name)
                state_type = 'inactive' if 'inactive' in well_file.name else 'active'
                
                # For well samples, we'll use committor values of 0.0 (inactive) or 1.0 (active)
                label_value = 0.0 if state_type == 'inactive' else 1.0
                
                # Extract sequence if available
                sequence = None
                if 'start_sequence' in data and data['start_sequence'] is not None:
                    sequence = data['start_sequence']
                elif 'end_sequence' in data and data['end_sequence'] is not None:
                    sequence = data['end_sequence']
                
                # Process each frame
                for i in range(len(data['positions'])):
                    # Generate unique frame ID
                    frame_id = f"well_{state_type}_{protein_name}_{frame_count:06d}"
                    while frame_id in used_frame_ids:
                        frame_count += 1
                        frame_id = f"well_{state_type}_{protein_name}_{frame_count:06d}"
                    used_frame_ids.add(frame_id)
                    frame_group = hdf.create_group(frame_id)
                    
                    # Store frame data - positions should already be in (L, 3, 3) format
                    # Check if positions have the correct shape
                    position_data = data['positions'][i]
                    if position_data.ndim != 3 or position_data.shape[1] != 3 or position_data.shape[2] != 3:
                        print(f"  Warning: Skipping frame with incorrect coordinate shape {position_data.shape}")
                        continue
                    frame_group.create_dataset('positions', data=position_data)
                    frame_group.create_dataset('label', data=label_value)  # Committor value
                    
                    # Store sequence information
                    if sequence is not None:
                        if isinstance(sequence, np.ndarray):
                            # If it's a numpy array, convert to string first
                            frame_group.create_dataset('sequence', data=np.bytes_(str(sequence)))
                        else:
                            frame_group.create_dataset('sequence', data=str(sequence).encode('utf-8'))
                    
                    # Store metadata
                    frame_group.attrs['source'] = 'well_sample'
                    frame_group.attrs['protein_name'] = protein_name
                    frame_group.attrs['state_type'] = state_type
                    frame_group.attrs['frame_index'] = i
                    
                    frame_count += 1
                    
            except Exception as e:
                print(f"Error processing {well_file}: {e}")
                continue
        
        # Process barrier samples
        print("\nProcessing barrier samples...")
        for barrier_file in tqdm(barrier_files, desc="Barriers"):
            try:
                # Load data
                data = load_npz_file(barrier_file)
                
                # Extract protein name and direction
                protein_name = extract_protein_name(barrier_file.name)
                direction = 'unknown'
                if 'forward' in barrier_file.name:
                    direction = 'forward'
                elif 'backward' in barrier_file.name:
                    direction = 'backward'
                
                # For barrier samples, we use the label from the metadata or final CV value
                label_value = 1.0  # Default to active state
                if 'label' in data:
                    label_value = float(data['label'])
                elif 'metadata' in data and 'label' in data['metadata'].item():
                    label_value = float(data['metadata'].item()['label'])
                elif 'cv_values' in data:
                    # Use final CV value to determine label
                    final_cv = data['cv_values'][-1] if len(data['cv_values']) > 0 else 0.5
                    label_value = 0.0 if final_cv < 0.5 else 1.0
                
                # Extract sequence if available
                sequence = None
                if 'start_sequence' in data and data['start_sequence'] is not None:
                    sequence = data['start_sequence']
                elif 'end_sequence' in data and data['end_sequence'] is not None:
                    sequence = data['end_sequence']
                
                # Process each frame
                for i in range(len(data['positions'])):
                    # Generate unique frame ID
                    frame_id = f"barrier_{direction}_{protein_name}_{frame_count:06d}"
                    while frame_id in used_frame_ids:
                        frame_count += 1
                        frame_id = f"barrier_{direction}_{protein_name}_{frame_count:06d}"
                    used_frame_ids.add(frame_id)
                    frame_group = hdf.create_group(frame_id)
                    
                    # Store frame data - positions should already be in (L, 3, 3) format
                    # Check if positions have the correct shape
                    position_data = data['positions'][i]
                    if position_data.ndim != 3 or position_data.shape[1] != 3 or position_data.shape[2] != 3:
                        print(f"  Warning: Skipping frame with incorrect coordinate shape {position_data.shape}")
                        continue
                    frame_group.create_dataset('positions', data=position_data)
                    frame_group.create_dataset('label', data=label_value)  # Committor value
                    
                    # Store sequence information
                    if sequence is not None:
                        if isinstance(sequence, np.ndarray):
                            # If it's a numpy array, convert to string first
                            frame_group.create_dataset('sequence', data=np.bytes_(str(sequence)))
                        else:
                            frame_group.create_dataset('sequence', data=str(sequence).encode('utf-8'))
                    
                    # Store metadata
                    frame_group.attrs['source'] = 'barrier_sample'
                    frame_group.attrs['protein_name'] = protein_name
                    frame_group.attrs['direction'] = direction
                    frame_group.attrs['frame_index'] = i
                    
                    frame_count += 1
                    
            except Exception as e:
                print(f"Error processing {barrier_file}: {e}")
                continue
        
        # Add metadata
        hdf.attrs['total_frames'] = frame_count
        hdf.attrs['transition_paths'] = len(path_files)
        hdf.attrs['well_samples'] = len(well_files)
        hdf.attrs['barrier_samples'] = len(barrier_files)
        hdf.attrs['creation_date'] = str(np.datetime64('now')).encode('utf-8')
        hdf.attrs['coordinate_format'] = 'backbone_N_CA_C'.encode('utf-8')  # Indicate backbone format
        hdf.attrs['coordinate_shape'] = '(L_residues, 3_atoms, 3_coordinates)'.encode('utf-8')
        
        print(f"\n✓ Created path atlas with {frame_count} frames")
        print(f"  Output file: {output_file}")
        print(f"  Transition paths: {len(path_files)} files")
        print(f"  Well samples: {len(well_files)} files")
        print(f"  Barrier samples: {len(barrier_files)} files")
        
        # Print unique proteins
        protein_names = set()
        for frame_id in hdf.keys():
            frame_group = hdf[frame_id]
            if 'protein_name' in frame_group.attrs:
                protein_names.add(frame_group.attrs['protein_name'])
        
        print(f"  Unique proteins: {len(protein_names)} ({', '.join(sorted(protein_names))})")
        
        # Verify coordinate format of a sample frame
        if frame_count > 0:
            sample_frame_id = list(hdf.keys())[0]
            sample_frame = hdf[sample_frame_id]
            if 'positions' in sample_frame:
                pos_shape = sample_frame['positions'].shape
                print(f"  Sample coordinate shape: {pos_shape}")
                if len(pos_shape) == 3 and pos_shape[1] == 3 and pos_shape[2] == 3:
                    print("  ✓ Coordinate format is correct for ESM3 (L, 3, 3)")
                else:
                    print("  ⚠ Coordinate format may not be correct for ESM3")
    
    return True


def main():
    # Configuration
    paths_dir = "data_processed/enhanced_paths_backbone"
    wells_dir = "data_processed/enhanced_well_samples_backbone"
    barriers_dir = "data_processed/barrier_samples"  # Keep using existing barrier samples
    output_file = "data_processed/path_atlas_backbone.h5"
    
    # Create the path atlas
    success = create_path_atlas_backbone(paths_dir, wells_dir, barriers_dir, output_file)
    
    if success:
        print(f"\n✓ Path atlas creation complete!")
        print(f"  Atlas file: {output_file}")
        return 0
    else:
        print(f"\n✗ Path atlas creation failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())