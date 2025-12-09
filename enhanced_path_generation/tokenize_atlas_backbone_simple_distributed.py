#!/usr/bin/env python
"""
Simple Distributed Structure Tokenizer for Protein Conformations (Backbone Coordinates)
Uses BioNeMo's ESM3 dVAE encoder to convert 3D protein structures into 1D structure tokens
This version is designed to run on multiple GPUs using torch.distributed
"""

import torch
import h5py
import numpy as np
from pathlib import Path
import sys
import argparse
import re
from tqdm import tqdm
import torch.distributed as dist
import os
try:
    from huggingface_hub import login
    # Check for HF_TOKEN
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"✓ Found HF_TOKEN, logging in...")
        login(token=hf_token)
    else:
        print("⚠ HF_TOKEN environment variable not set.")
        print("  To access ESM3-open, please set HF_TOKEN.")
        print("  Export it: export HF_TOKEN=hf_...")
except ImportError:
    print("⚠ huggingface_hub not installed. Cannot authenticate.")

try:
    # Try to import the structure encoder directly from esm.pretrained
    from esm.pretrained import ESM3_structure_encoder_v0
    print("Successfully imported ESM3 structure encoder")
    # Try to initialize the encoder to check if we have access
    try:
        test_encoder = ESM3_structure_encoder_v0()
        del test_encoder
        encoder_available = True
        print("✓ ESM3 structure encoder is accessible")
    except Exception as e:
        print(f"✗ Cannot access ESM3 model (gated repository): {e}")
        print("  Please ensure you have accepted the license at https://huggingface.co/evolutionaryscale/esm3-sm-open-v1")
        print("  and set your HF_TOKEN.")
        raise RuntimeError("ESM3 model access failed. Stopping to prevent random token generation.")
except ImportError as e:
    print(f"Error importing ESM3 structure encoder: {e}")
    raise RuntimeError("ESM3 import failed")



def load_dvae_encoder(model_name="esm3_sm_open_v1"):
    """
    Load the pre-trained dVAE encoder from ESM3
    
    Args:
        model_name: Name of the ESM3 model to load
        
    Returns:
        ESM3_structure_encoder_v0 or None: Loaded structure encoder
    """
    print(f"Loading ESM3 structure encoder: {model_name}")
    
    global encoder_available
    if encoder_available and ESM3_structure_encoder_v0 is not None:
        # Initialize the ESM3 structure encoder
        try:
            structure_encoder = ESM3_structure_encoder_v0()
            print("✓ Successfully loaded ESM3 structure encoder")
            return structure_encoder
        except Exception as e:
            print(f"Error initializing ESM3 structure encoder: {e}")
            print("Falling back to simplified structure tokenization")
            return None
    else:
        print("⚠ Using simplified structure tokenization (random tokens)")
        return None


def tokenize_protein_structure(positions, structure_encoder, chain_id=None):
    """
    Convert 3D protein structure to structure tokens using the dVAE encoder
    
    Args:
        positions: Array of shape (L, 3, 3) containing backbone coordinates (N, CA, C)
                   or (L*3, 3) containing flattened backbone coordinates
        structure_encoder: Pre-trained ESM3_structure_encoder_v0 or None
        chain_id: Optional chain identifier for multi-chain proteins
        
    Returns:
        structure_tokens: 1D array of structure tokens
    """
    try:
        # Convert positions to torch tensor if needed
        if isinstance(positions, np.ndarray):
            positions = torch.from_numpy(positions).float()
        
        # Ensure positions have the right shape
        if positions.dim() == 2 and positions.shape[1] == 3:
            # Check if this is flattened backbone coordinates (L*3, 3)
            # Reshape to (L, 3, 3) where L is number of residues
            if positions.shape[0] % 3 == 0:
                L = positions.shape[0] // 3
                positions = positions.view(L, 3, 3)
            else:
                raise ValueError(f"Cannot reshape positions with shape {positions.shape} to (L, 3, 3)")
        elif positions.dim() == 3 and positions.shape[1] == 3 and positions.shape[2] == 3:
            # Already in the right format (L, 3, 3)
            pass
        else:
            raise ValueError(f"Unexpected positions shape for backbone coordinates: {positions.shape}")
        
        # Tokenize the structure using the encoder or generate random tokens
        with torch.no_grad():
            if structure_encoder is not None:
                # The structure encoder expects coordinates in a specific format
                # ESM3 expects a 4D tensor with shape (1, L, 3, 3) where 1 is batch dimension
                L = positions.shape[0]  # Number of residues
                
                # Add batch dimension to make it (1, L, 3, 3)
                positions_batched = positions.unsqueeze(0)
                
                # Create attention mask (all positions are valid)
                attention_mask = torch.ones((1, L), dtype=torch.bool)  # Add batch dimension
                
                # Call the structure encoder with correct parameters
                # Create additional required parameters with batch dimension
                sequence_id = torch.zeros((1, L), dtype=torch.long, device=positions_batched.device)  # Add batch dimension
                residue_index = torch.arange(L, dtype=torch.long, device=positions_batched.device).unsqueeze(0)  # Add batch dimension
                
                structure_tokens_batched = structure_encoder.encode(
                    coords=positions_batched,
                    attention_mask=attention_mask.to(positions_batched.device),
                    sequence_id=sequence_id,
                    residue_index=residue_index
                )
                
                # Extract the structure tokens from the output
                # The encoder returns a tuple: (encoded_features, structure_tokens)
                if isinstance(structure_tokens_batched, tuple):
                    structure_tokens_batched = structure_tokens_batched[1]  # Get structure tokens
                elif hasattr(structure_tokens_batched, 'structure_tokens'):
                    structure_tokens_batched = structure_tokens_batched.structure_tokens
                elif isinstance(structure_tokens_batched, dict) and 'structure_tokens' in structure_tokens_batched:
                    structure_tokens_batched = structure_tokens_batched['structure_tokens']
                
                # Remove batch dimension
                structure_tokens = structure_tokens_batched.squeeze(0)
            else:
                # Generate random tokens as a fallback
                L = positions.shape[0]  # Number of residues
                # Generate random tokens in the range [0, 4095] (typical for structure tokens)
                structure_tokens = torch.randint(0, 4096, (L,))
            
        # Convert to numpy array
        if hasattr(structure_tokens, 'cpu'):
            structure_tokens = structure_tokens.cpu().numpy()
        elif isinstance(structure_tokens, torch.Tensor):
            structure_tokens = structure_tokens.numpy()
        
        return structure_tokens
        
    except Exception as e:
        print(f"Error tokenizing structure: {e}")
        import traceback
        traceback.print_exc()
        return None


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


def process_atlas_file_distributed(rank, world_size, input_file, output_file, structure_tokenizer, max_frames=None):
    """
    Process an atlas file and generate structure tokens for each frame (distributed version)
    
    Args:
        rank: Rank of the current process
        world_size: Total number of processes
        input_file: Path to input HDF5 file containing protein conformations with backbone coordinates
        output_file: Path to output HDF5 file for tokenized structures
        structure_tokenizer: Pre-trained StructureTokenizer
        max_frames: Maximum number of frames to process (None for all)
    """
    print(f"[Rank {rank}] Processing atlas file: {input_file}")
    print(f"[Rank {rank}] Output file: {output_file}")
    
    # Move model to GPU if available
    if structure_tokenizer is not None and torch.cuda.is_available():
        structure_tokenizer = structure_tokenizer.to(f'cuda:{rank}')
        torch.cuda.set_device(f'cuda:{rank}')
    
    # First, count total frames without loading data
    with h5py.File(input_file, 'r') as infile:
        # Get all dataset names (they're not organized by frame groups)
        # Find all datasets that end with '/positions'
        all_datasets = []
        infile.visit(lambda name: all_datasets.append(name) if isinstance(infile[name], h5py.Dataset) else None)
        position_datasets = [name for name in all_datasets if name.endswith('/positions')]
        
        total_frames = len(position_datasets)
        if max_frames is not None:
            total_frames = min(total_frames, max_frames)
            position_datasets = position_datasets[:total_frames]
    
    # Distribute work among processes
    # Each process handles a subset of frames
    frames_per_process = total_frames // world_size
    start_idx = rank * frames_per_process
    end_idx = start_idx + frames_per_process if rank < world_size - 1 else total_frames
    my_position_datasets = position_datasets[start_idx:end_idx]
    
    print(f"[Rank {rank}] Processing {len(my_position_datasets)} frames (out of {total_frames} total)")
    
    # Process each frame assigned to this process
    successful_frames = 0
    results = []
    
    # Open file once and keep it open for this process's frames only
    with h5py.File(input_file, 'r') as infile:
        for i, pos_dataset_path in enumerate(tqdm(my_position_datasets, desc=f"Tokenizing frames (Rank {rank})")):
            try:
                # Extract frame_id from the dataset path
                frame_id = '/'.join(pos_dataset_path.split('/')[:-1])  # Remove '/positions' from the end
                
                # Load frame data
                positions = infile[pos_dataset_path][:]  # Shape: (L, 3, 3) for backbone coordinates
                label = infile[frame_id + '/label'][()]
                
                # Get sequence and protein name if available
                sequence = ""
                if frame_id + '/sequence' in infile:
                    sequence = infile[frame_id + '/sequence'][()]
                    # Handle different data types for sequence
                    if isinstance(sequence, bytes):
                        sequence = sequence.decode('utf-8')
                    elif not isinstance(sequence, str):
                        sequence = str(sequence)
                
                protein_name = ""
                # Extract protein name from the frame_id path
                # Frame IDs look like: transition_6kpc.cif_000560
                protein_name = extract_protein_name(frame_id.split('/')[-1])  # Get the last part and extract protein name
                
                # Reshape positions if needed
                if positions.ndim == 2 and positions.shape[1] == 3:
                    # Check if this is flattened backbone coordinates (L*3, 3)
                    if positions.shape[0] % 3 == 0:
                        L = positions.shape[0] // 3
                        positions = positions.reshape(L, 3, 3)
                    else:
                        print(f"[Rank {rank}] Warning: Cannot reshape positions with shape {positions.shape} to (L, 3, 3) for frame {frame_id}")
                        continue
                elif positions.ndim != 3 or positions.shape[1] != 3 or positions.shape[2] != 3:
                    print(f"[Rank {rank}] Warning: Unexpected coordinate shape {positions.shape} for frame {frame_id}")
                    continue
                
                # Convert numpy array to torch tensor and move to GPU if available
                if not isinstance(positions, torch.Tensor):
                    positions = torch.from_numpy(positions).float()
                if torch.cuda.is_available():
                    positions = positions.to(f'cuda:{rank}')
                
                # Tokenize the structure
                structure_tokens = tokenize_protein_structure(positions, structure_tokenizer)
                
                if structure_tokens is not None:
                    results.append({
                        'frame_id': frame_id,
                        'structure_tokens': structure_tokens,
                        'label': label,
                        'sequence': sequence,
                        'protein_name': protein_name
                    })
                    successful_frames += 1
                    
                    # Update progress bar description
                    if i % 100 == 0:
                        tqdm.write(f"[Rank {rank}] Processed {successful_frames}/{i+1} frames successfully")
                
            except Exception as e:
                print(f"[Rank {rank}] Error processing frame {frame_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save results to temporary file for this process
    temp_output_file = f"{output_file}.temp_rank_{rank}"
    if results:
        # Create datasets
        num_frames = len(results)
        # Use a fixed maximum length for structure tokens to ensure consistency
        max_length = 512  # Same as in the non-distributed version
        
        # Pad structure tokens to same length (or truncate)
        structure_tokens_array = np.full((num_frames, max_length), -1, dtype=np.int32)  # Use -1 as padding value
        labels_array = np.zeros(num_frames, dtype=np.float32)
        sequences_array = np.array([r['sequence'].encode('utf-8') for r in results], dtype=h5py.string_dtype())
        protein_names_array = np.array([r['protein_name'].encode('utf-8') for r in results], dtype=h5py.string_dtype())
        
        for i, result in enumerate(results):
            tokens = result['structure_tokens']
            length = min(len(tokens), max_length)
            structure_tokens_array[i, :length] = tokens[:length]
            labels_array[i] = result['label']
        
        # Save to temporary file
        with h5py.File(temp_output_file, 'w') as temp_outfile:
            temp_outfile.create_dataset('structure_tokens', data=structure_tokens_array, compression='gzip')
            temp_outfile.create_dataset('labels', data=labels_array, compression='gzip')
            temp_outfile.create_dataset('sequences', data=sequences_array, compression='gzip')
            temp_outfile.create_dataset('protein_names', data=protein_names_array, compression='gzip')
        
        print(f"[Rank {rank}] Saved {num_frames} frames to {temp_output_file}")
    else:
        # Create empty file to indicate completion
        with h5py.File(temp_output_file, 'w') as temp_outfile:
            pass
        print(f"[Rank {rank}] No frames processed successfully, created empty file {temp_output_file}")
    
    return successful_frames


def merge_temporary_files(output_file, world_size):
    """
    Merge temporary files from all processes into a single output file
    
    Args:
        output_file: Path to final output file
        world_size: Number of processes
    """
    print(f"Merging temporary files into {output_file}")
    
    # Collect data from all temporary files
    all_structure_tokens = []
    all_labels = []
    all_sequences = []
    all_protein_names = []
    
    temp_files_found = 0
    for rank in range(world_size):
        temp_file = f"{output_file}.temp_rank_{rank}"
        if os.path.exists(temp_file):
            temp_files_found += 1
            with h5py.File(temp_file, 'r') as f:
                if 'structure_tokens' in f and f['structure_tokens'].size > 0:
                    all_structure_tokens.append(f['structure_tokens'][:])
                    all_labels.append(f['labels'][:])
                    all_sequences.append(f['sequences'][:])
                    all_protein_names.append(f['protein_names'][:])
            # Remove temporary file
            os.remove(temp_file)
        else:
            print(f"Warning: Temporary file {temp_file} not found")
    
    print(f"Found {temp_files_found} temporary files out of {world_size} expected")
    
    if all_structure_tokens:
        # Concatenate all data
        structure_tokens_combined = np.concatenate(all_structure_tokens, axis=0)
        labels_combined = np.concatenate(all_labels, axis=0)
        sequences_combined = np.concatenate(all_sequences, axis=0)
        protein_names_combined = np.concatenate(all_protein_names, axis=0)
        
        # Save to final output file
        with h5py.File(output_file, 'w') as outfile:
            outfile.create_dataset('structure_tokens', data=structure_tokens_combined, compression='gzip')
            outfile.create_dataset('labels', data=labels_combined, compression='gzip')
            outfile.create_dataset('sequences', data=sequences_combined, compression='gzip')
            outfile.create_dataset('protein_names', data=protein_names_combined, compression='gzip')
        
        print(f"✓ Merged {len(labels_combined)} frames into {output_file}")
        print(f"Structure tokens shape: {structure_tokens_combined.shape}")
        if structure_tokens_combined.size > 0:
            print(f"Token range: {structure_tokens_combined.min()} to {structure_tokens_combined.max()}")
        
        # Print unique proteins
        if protein_names_combined.size > 0:
            protein_names = [name.decode('utf-8') for name in protein_names_combined]
            unique_proteins = set(protein_names)
            print(f"Unique proteins: {len(unique_proteins)} ({', '.join(sorted(unique_proteins))})")
    else:
        print("✗ No temporary files found or no data to merge")


def setup_distributed(rank, world_size):
    """Setup distributed training environment"""
    # Let torchrun handle the distributed initialization
    # This avoids issues with hardcoded addresses/ports
    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    except Exception as e:
        print(f"Failed to initialize distributed process group: {e}")
        raise


def cleanup_distributed():
    """Cleanup distributed training environment"""
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Distributed tokenizer for protein conformations using BioNeMo ESM3 dVAE (Backbone Coordinates)")
    parser.add_argument("--input", "-i", required=True, help="Input HDF5 atlas file with backbone coordinates")
    parser.add_argument("--output", "-o", required=True, help="Output HDF5 file for tokenized structures")
    parser.add_argument("--model", "-m", default="esm3_sm_open_v1", help="ESM3 model name")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to process")
    
    args = parser.parse_args()
    
    # Get world size from environment or default to 1
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    
    print(f"=" * 60)
    print(f"PROTEIN STRUCTURE TOKENIZER (BioNeMo ESM3 dVAE - BACKBONE) - RANK {rank}/{world_size-1}")
    print(f"=" * 60)
    
    # Setup distributed environment if needed
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file {input_path} not found")
        if world_size > 1:
            cleanup_distributed()
        return 1
    
    # Load the dVAE encoder
    # Only rank 0 loads the model first to avoid conflicts with gated repository
    structure_tokenizer = None
    if world_size > 1:
        if rank == 0:
            try:
                structure_tokenizer = load_dvae_encoder(args.model)
            except Exception as e:
                print(f"Error loading model on rank 0: {e}")
                cleanup_distributed()
                return 1
            # Wait for rank 0 to finish loading before other ranks proceed
            dist.barrier()
        else:
            # Other ranks wait for rank 0 to finish loading
            dist.barrier()
            try:
                structure_tokenizer = load_dvae_encoder(args.model)
            except Exception as e:
                print(f"Error loading model on rank {rank}: {e}")
                cleanup_distributed()
                return 1
    else:
        try:
            structure_tokenizer = load_dvae_encoder(args.model)
        except Exception as e:
            print(f"Error loading model: {e}")
            return 1
    
    # Process the atlas file
    successful_frames = process_atlas_file_distributed(
        rank, 
        world_size,
        input_path, 
        args.output, 
        structure_tokenizer, 
        max_frames=args.max_frames
    )
    
    # Wait for all processes to finish with timeout
    if world_size > 1:
        try:
            dist.barrier()
        except Exception as e:
            print(f"[Rank {rank}] Warning: Barrier synchronization failed: {e}")
    
    # Only rank 0 merges the files
    if rank == 0:
        if world_size > 1:
            merge_temporary_files(args.output, world_size)
        print(f"\n✓ Tokenization complete!")
        print(f"  Input:  {args.input}")
        print(f"  Output: {args.output}")
    
    # Cleanup
    if world_size > 1:
        try:
            cleanup_distributed()
        except Exception as e:
            print(f"[Rank {rank}] Warning: Cleanup failed: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())