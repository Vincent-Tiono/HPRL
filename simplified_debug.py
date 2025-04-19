#!/usr/bin/env python
"""
Simplified debug script to show the shapes of tensors in the ActionBehaviorEncoder.

This script directly loads an HDF5 file and feeds it to get_exec_data without using make_datasets.
"""

import torch
import h5py
import numpy as np
import sys
import os
import logging

# Add project root to Python path
sys.path.insert(0, '.')

from pretrain.models_option_new_vae import ActionBehaviorEncoder
from karel_env.dsl import get_DSL_option_v2

def get_exec_data(hdf5_file, program_id, num_agent_actions):
    """Extract execution data from HDF5 file for a given program."""
    # if program_id not in hdf5_file or 's_h' not in hdf5_file[program_id]:
    #     return np.array([]), np.array([]), np.array([]), np.array([])
        
    s_h = np.moveaxis(np.copy(hdf5_file[program_id]['s_h']), [-1,-2,-3], [-3,-1,-2])
    a_h = np.copy(hdf5_file[program_id]['a_h'])
    s_h_len = np.copy(hdf5_file[program_id]['s_h_len'])
    a_h_len = np.copy(hdf5_file[program_id]['a_h_len'])
    
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
    
    return s_h, s_h_len, a_h, a_h_len

def debug_tensor_shapes():
    # Path to HDF5 file
    hdf5_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/data.hdf5"
    
    # Initialize DSL
    dsl = get_DSL_option_v2(seed=42)
    num_agent_actions = len(dsl.action_functions) + 1  # +1 for no-op
    
    print(f"Number of agent actions: {num_agent_actions}")
    
    # Setup logger
    logger = logging.getLogger('simplified_debug')
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    
    # Process the HDF5 file - directly open it without using make_datasets
    try:
        print(f"Opening HDF5 file: {hdf5_file_path}")
        hdf5_file = h5py.File(hdf5_file_path, 'r')
            
        # Get a sample program ID
        sample_keys = [k for k in hdf5_file.keys() if not k.startswith('data_info_')]
        
        if not sample_keys:
            print("No program samples found in the HDF5 file")
            return
            
        # Get the first program ID
        program_id = sample_keys[5]
        print(f"Using program ID: {program_id}")
        
        # Extract execution data directly from the HDF5 file
        s_h, s_h_len, a_h, a_h_len = get_exec_data(hdf5_file, program_id, num_agent_actions)
        
        # Print numpy array shapes before conversion to tensors
        print("\nNumPy array shapes before conversion:")
        print(f"s_h.shape: {s_h.shape}")  # Should be (num_demos, T, C, H, W)
        print(f"s_h_len.shape: {s_h_len.shape}")  # Should be (num_demos,)
        print(f"a_h.shape: {a_h.shape}")  # Should be (num_demos, T)
        print(f"a_h_len.shape: {a_h_len.shape}")  # Should be (num_demos,)
        
        # Convert to PyTorch tensors
        s_h_tensor = torch.tensor(s_h, dtype=torch.float32)
        s_h_len_tensor = torch.tensor(s_h_len, dtype=torch.int16)
        a_h_tensor = torch.tensor(a_h, dtype=torch.int16)
        a_h_len_tensor = torch.tensor(a_h_len, dtype=torch.int16)
        
        # Print tensor shapes
        print("\nPyTorch tensor shapes after conversion:")
        print(f"s_h_tensor.shape: {s_h_tensor.shape}")  # Should be (num_demos, T, C, H, W)
        print(f"s_h_len_tensor.shape: {s_h_len_tensor.shape}")  # Should be (num_demos,)
        print(f"a_h_tensor.shape: {a_h_tensor.shape}")  # Should be (num_demos, T)
        print(f"a_h_len_tensor.shape: {a_h_len_tensor.shape}")  # Should be (num_demos,)
        
        # Add batch dimension and check shapes as they would be in ActionBehaviorEncoder.forward
        s_h_batch = s_h_tensor.unsqueeze(0)  # Add batch dimension (B=1)
        a_h_batch = a_h_tensor.unsqueeze(0)  # Add batch dimension (B=1)
        s_h_len_batch = s_h_len_tensor.unsqueeze(0)  # Add batch dimension (B=1)
        a_h_len_batch = a_h_len_tensor.unsqueeze(0)  # Add batch dimension (B=1)
        
        print("\nPyTorch tensor shapes with batch dimension added:")
        print(f"s_h_batch.shape: {s_h_batch.shape}")  # Should be (B, num_demos, T, C, H, W)
        print(f"a_h_batch.shape: {a_h_batch.shape}")  # Should be (B, num_demos, T)
        print(f"s_h_len_batch.shape: {s_h_len_batch.shape}")  # Should be (B, num_demos)
        print(f"a_h_len_batch.shape: {a_h_len_batch.shape}")  # Should be (B, num_demos)
        
        # Initialize the ActionBehaviorEncoder with appropriate parameters
        encoder_config = {
            'recurrent_policy': True,
            'input_channel': 8,
            'input_height': 8,
            'input_width': 8,
            'fuse_s_0': False
        }
        
        encoder = ActionBehaviorEncoder(
            recurrent=True,
            num_actions=num_agent_actions,
            hidden_size=64,
            rnn_type='GRU',
            dropout=0.0,
            use_linear=True,
            unit_size=256,
            **encoder_config
        )
        
        # Check expected input shapes for the encoder
        print("\nExpected input shapes for ActionBehaviorEncoder.forward:")
        print("s_h: shape (B, R, T, C, H, W)") 
        print("a_h: shape (B, R, T)")
        print("s_h_len: shape (B, R)")
        print("a_h_len: shape (B, R)")
        
        # Fix shapes if needed for encoder
        if len(s_h_batch.shape) == 4:  # If shape is (B, num_demos, T, features)
            s_h_batch = s_h_batch.unsqueeze(2)  # Add T dimension if missing
        
        print("\nActual shapes being passed to the encoder:")
        print(f"s_h_batch.shape: {s_h_batch.shape}")
        print(f"a_h_batch.shape: {a_h_batch.shape}")
        print(f"s_h_len_batch.shape: {s_h_len_batch.shape}")
        print(f"a_h_len_batch.shape: {a_h_len_batch.shape}")
        
        # Close the HDF5 file
        hdf5_file.close()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_tensor_shapes()