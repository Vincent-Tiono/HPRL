#!/usr/bin/env python
"""
Debug script to show the shapes of tensors in the ActionBehaviorEncoder.

This script loads a sample from an HDF5 file and prints the shapes of
key tensors used in the encoder.
"""

import torch
import h5py
import numpy as np
import sys
import os
from tqdm import tqdm
import logging
import traceback

# Add project root to Python path
sys.path.insert(0, '.')

from pretrain.models_option_new_vae import ActionBehaviorEncoder
from karel_env.dsl import get_DSL_option_v2

def get_exec_data(hdf5_file, program_id, num_agent_actions):
    """Extract execution data from HDF5 file for a given program."""
    print(f"Extracting data for program {program_id}")
    
    s_h = np.moveaxis(np.copy(hdf5_file[program_id]['s_h']), [-1,-2,-3], [-3,-1,-2])
    a_h = np.copy(hdf5_file[program_id]['a_h'])
    s_h_len = np.copy(hdf5_file[program_id]['s_h_len'])
    a_h_len = np.copy(hdf5_file[program_id]['a_h_len'])
    
    print(f"Raw shapes from HDF5:")
    print(f"s_h.shape: {s_h.shape}")
    print(f"s_h_len.shape: {s_h_len.shape}")
    print(f"a_h.shape: {a_h.shape}")
    print(f"a_h_len.shape: {a_h_len.shape}")
    
    if s_h.shape[1] == 1:
        s_h = np.concatenate((s_h, s_h), axis=1)
        a_h = np.ones((s_h.shape[0], 1))
        
    for i in range(s_h_len.shape[0]):
        if a_h_len[i] == 0:
            assert s_h_len[i] == 1
            a_h_len[i] += 1
            s_h_len[i] += 1
            s_h[i][1] = s_h[i][0]
            a_h[i][0] = num_agent_actions - 1
    
    print(f"Processed shapes after adjustments:")
    print(f"s_h.shape: {s_h.shape}")
    print(f"s_h_len.shape: {s_h_len.shape}")
    print(f"a_h.shape: {a_h.shape}")
    print(f"a_h_len.shape: {a_h_len.shape}")
    
    return s_h, s_h_len, a_h, a_h_len

def debug_tensor_shapes():
    # Path to HDF5 file
    hdf5_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/data.hdf5"
    
    # Initialize DSL
    dsl = get_DSL_option_v2(seed=42)
    num_agent_actions = len(dsl.action_functions) + 1  # +1 for no-op
    
    print(f"Number of agent actions: {num_agent_actions}")
    
    try:
        # Open the HDF5 file
        with h5py.File(hdf5_file_path, 'r') as hdf5_file:
            # Get a sample program ID
            sample_keys = [k for k in hdf5_file.keys() if not k.startswith('data_info_')]
            
            if not sample_keys:
                print("No program samples found in the HDF5 file")
                return
                
            # Get the first program ID
            program_id = sample_keys[0]
            print(f"Using program ID: {program_id}")
            
            # Extract execution data
            s_h, s_h_len, a_h, a_h_len = get_exec_data(hdf5_file, program_id, num_agent_actions)
            
            # Convert to PyTorch tensors
            s_h_tensor = torch.tensor(s_h, dtype=torch.float32)
            s_h_len_tensor = torch.tensor(s_h_len, dtype=torch.int64)
            a_h_tensor = torch.tensor(a_h, dtype=torch.int64)
            a_h_len_tensor = torch.tensor(a_h_len, dtype=torch.int64)
            
            print("\nPyTorch tensor shapes:")
            print(f"s_h_tensor.shape: {s_h_tensor.shape}")
            print(f"s_h_len_tensor.shape: {s_h_len_tensor.shape}")
            print(f"a_h_tensor.shape: {a_h_tensor.shape}")
            print(f"a_h_len_tensor.shape: {a_h_len_tensor.shape}")
            
            # Test 1: Add batch dimension to match ActionBehaviorEncoder's expected input
            s_h_batch = s_h_tensor.unsqueeze(0)
            a_h_batch = a_h_tensor.unsqueeze(0)
            s_h_len_batch = s_h_len_tensor.unsqueeze(0)
            a_h_len_batch = a_h_len_tensor.unsqueeze(0)
            
            print("\nWith batch dimension (B=1) added:")
            print(f"s_h_batch.shape: {s_h_batch.shape}")
            print(f"a_h_batch.shape: {a_h_batch.shape}")
            print(f"s_h_len_batch.shape: {s_h_len_batch.shape}")
            print(f"a_h_len_batch.shape: {a_h_len_batch.shape}")
            
            # Initialize the ActionBehaviorEncoder
            encoder = ActionBehaviorEncoder(
                recurrent=True,
                num_actions=num_agent_actions,
                hidden_size=64,
                rnn_type='GRU',
                dropout=0.0,
                use_linear=True,
                unit_size=256,
                input_channel=s_h.shape[2],
                input_height=s_h.shape[3],
                input_width=s_h.shape[4],
                fuse_s_0=False
            )
            
            # Reshape s_h to match the encoder's expected input format if needed
            # According to the forward method, s_h should be (B, R, 1, C, H, W)
            # where 1 is for the initial state only
            if len(s_h_batch.shape) == 5:  # If shape is (B, R, C, H, W)
                # Add the timestep dimension for initial state
                s_h_initial = s_h_batch[:, :, 0:1, :, :]
                print(f"\nReshaping for encoder - initial state only:")
                print(f"s_h_initial.shape: {s_h_initial.shape}")  # Should be (B, R, 1, C, H, W)
            
            print("\nExpected shapes for ActionBehaviorEncoder.forward:")
            print("s_h: shape (B, R, 1, C, H, W) - B=batch, R=num_demos, 1=initial timestep only")
            print("a_h: shape (B, R, T) - T=action sequence length")
            print("s_h_len: shape (B, R)")
            print("a_h_len: shape (B, R)")
            
            # Test with multiple programs in a batch
            # Simulate a batch of 3 programs
            s_h_multi = torch.cat([s_h_batch, s_h_batch, s_h_batch], dim=0)
            a_h_multi = torch.cat([a_h_batch, a_h_batch, a_h_batch], dim=0)
            s_h_len_multi = torch.cat([s_h_len_batch, s_h_len_batch, s_h_len_batch], dim=0)
            a_h_len_multi = torch.cat([a_h_len_batch, a_h_len_batch, a_h_len_batch], dim=0)
            
            print("\nWith multiple programs in batch (B=3):")
            print(f"s_h_multi.shape: {s_h_multi.shape}")
            print(f"a_h_multi.shape: {a_h_multi.shape}")
            print(f"s_h_len_multi.shape: {s_h_len_multi.shape}")
            print(f"a_h_len_multi.shape: {a_h_len_multi.shape}")
            
            # Create s_h_initial for the multi-batch case
            if len(s_h_multi.shape) == 5:  # If shape is (B, R, C, H, W)
                initial_states = []
                for i in range(s_h_multi.shape[0]):
                    # For each program, take only the first state from each demo
                    initial_state = s_h_multi[i, :, 0:1, :, :]
                    initial_states.append(initial_state)
                s_h_initial_multi = torch.stack(initial_states, dim=0)
                print(f"\nInitial states for multi-batch:")
                print(f"s_h_initial_multi.shape: {s_h_initial_multi.shape}")  # Should be (B, R, 1, C, H, W)
            
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_shapes()