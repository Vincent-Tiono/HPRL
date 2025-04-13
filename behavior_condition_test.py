import os
import sys
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from pretrain.models_option_new_vae import VAE, ConditionPolicy, ProgramVAE, BehaviorEncoder
from karel_env.dsl import get_DSL_option_v2

# Simple behavior encoder for our test
class SimpleBehaviorEncoder(nn.Module):
    def __init__(self, state_shape, action_dim, hidden_dim=64, latent_dim=64):
        super(SimpleBehaviorEncoder, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Encoder for states
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoder
        self.action_encoder = nn.Embedding(action_dim, action_dim)
        
        # RNN for sequential processing
        self.rnn = nn.GRU(hidden_dim + action_dim, hidden_dim, batch_first=True)
        
        # Latent projection
        self.latent_mean = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)
    
    def forward(self, s_h, a_h, a_h_len):
        """
        Args:
            s_h: Tensor of shape [B, num_demos, max_demo_len, C, H, W]
            a_h: Tensor of shape [B, num_demos, max_demo_len]
            a_h_len: Tensor of shape [B, num_demos]
        """
        # Get dimensions
        batch_size, num_demos, max_demo_len = s_h.shape[0], s_h.shape[1], s_h.shape[2]
        C, H, W = s_h.shape[3], s_h.shape[4], s_h.shape[5]
        
        # Debug print to check actual shapes
        print(f"Shape of s_h: {s_h.shape}")
        print(f"Shape of a_h: {a_h.shape}")
        print(f"Shape of a_h_len: {a_h_len.shape}")
        
        # The issue: s_h has shape [B, num_demos, 51, C, H, W] but a_h has [B, num_demos, 50]
        # We need to trim s_h or pad a_h to make them match
        
        # Option 1: Trim s_h to match a_h's length (simpler approach)
        s_h = s_h[:, :, :a_h.shape[2], :, :, :]
        print(f"Trimmed s_h shape: {s_h.shape}")
        
        # Reshape with explicit size verification
        expected_elements = batch_size * num_demos * a_h.shape[2] * C * H * W
        actual_elements = s_h.numel()
        
        if expected_elements != actual_elements:
            print(f"WARNING: Expected {expected_elements} elements, got {actual_elements}")
            # Fallback to safer reshape
            s_h_flattened = s_h.reshape(-1, C, H, W)
        else:
            s_h_flattened = s_h.reshape(batch_size * num_demos * a_h.shape[2], C, H, W)
        
        print(f"Flattened s_h shape: {s_h_flattened.shape}")
        
        # Encode states
        state_emb = self.state_encoder(s_h_flattened)
        
        # Safely reshape back to batch structure
        state_emb = state_emb.reshape(batch_size, num_demos, a_h.shape[2], -1)
        print(f"State embedding shape: {state_emb.shape}")
        
        # Encode actions (shift by 1 to get the previous action for each state)
        a_h_shifted = torch.cat([
            torch.full((batch_size, num_demos, 1), self.action_dim-1, device=a_h.device, dtype=a_h.dtype),
            a_h[:, :, :-1]
        ], dim=2)
        action_emb = self.action_encoder(a_h_shifted)
        print(f"Action embedding shape: {action_emb.shape}")
        
        # Now dimensions should match for concatenation
        combined = torch.cat([state_emb, action_emb], dim=-1)
        print(f"Combined embedding shape: {combined.shape}")
        
        # Process through RNN
        combined = combined.reshape(batch_size * num_demos, a_h.shape[2], -1)
        a_h_len = a_h_len.reshape(batch_size * num_demos)
        
        # Pack sequence for RNN processing
        packed = rnn.pack_padded_sequence(
            combined, a_h_len.cpu(), batch_first=True, enforce_sorted=False
        )
        
        _, h_n = self.rnn(packed)
        h_n = h_n.reshape(batch_size, num_demos, -1).mean(dim=1)  # Average across demos
        
        # Project to latent space
        mean = self.latent_mean(h_n)
        logvar = self.latent_logvar(h_n)
        
        # Sample using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        
        return z, mean, logvar

# Simple Condition Policy 
class SimpleConditionPolicy(nn.Module):
    def __init__(self, state_shape, action_dim, latent_dim=64, hidden_dim=64):
        super(SimpleConditionPolicy, self).__init__()
        self.state_shape = state_shape
        self.action_dim = action_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Conv2d(state_shape[0], 32, 3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoder for previous actions
        self.action_encoder = nn.Embedding(action_dim, action_dim)
        
        # RNN for sequential processing
        self.rnn = nn.GRU(hidden_dim + action_dim + latent_dim, hidden_dim, batch_first=True)
        
        # Action predictor
        self.action_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, s_h, a_h, b_z, deterministic=True):
        """
        Args:
            s_h: Tensor of shape [B, num_demos, max_demo_len, C, H, W]
            a_h: Tensor of shape [B, num_demos, max_demo_len]
            b_z: Behavior embeddings tensor of shape [B, latent_dim]
        
        Returns:
            action_logits of shape [batch_size, num_demos, seq_len, num_actions]
        """
        # Get dimensions
        batch_size, num_demos = s_h.shape[0], s_h.shape[1]
        C, H, W = s_h.shape[3], s_h.shape[4], s_h.shape[5]
        
        # Debug prints
        print(f"ConditionPolicy - s_h shape: {s_h.shape}")
        print(f"ConditionPolicy - a_h shape: {a_h.shape}")
        print(f"ConditionPolicy - b_z shape: {b_z.shape}")
        
        # Trim s_h to match a_h's sequence length
        s_h = s_h[:, :, :a_h.shape[2], :, :, :]
        print(f"ConditionPolicy - trimmed s_h shape: {s_h.shape}")
        
        # Create masks for valid actions
        action_masks = (a_h != self.action_dim - 1).unsqueeze(-1).float()
        
        # Initialize action logits storage
        action_logits_all = []
        
        # Encode initial states
        init_states = s_h[:, :, 0].reshape(batch_size * num_demos, C, H, W)
        state_emb = self.state_encoder(init_states)
        state_emb = state_emb.reshape(batch_size, num_demos, -1)
        
        # Expand behavior embedding for each demo
        b_z_expanded = b_z.unsqueeze(1).expand(-1, num_demos, -1)
        
        # Initial action (start token)
        actions = torch.full(
            (batch_size, num_demos, 1), 
            self.action_dim - 1, 
            device=s_h.device, 
            dtype=torch.long
        )
        
        # Combine for initial rnn hidden state
        hidden = torch.zeros(
            1, batch_size * num_demos, state_emb.shape[-1], 
            device=s_h.device
        )
        
        # Process each step in the sequence
        seq_length = a_h.shape[2] - 1  # -1 because we predict next actions
        for i in range(seq_length):
            # Get action embedding for previous step
            action_emb = self.action_encoder(
                actions[:, :, -1]
            ).reshape(batch_size, num_demos, -1)
            
            # Combine state, action, and behavior embeddings
            inputs = torch.cat([
                state_emb, 
                action_emb, 
                b_z_expanded
            ], dim=-1)
            
            # Process through RNN
            inputs = inputs.reshape(batch_size * num_demos, 1, -1)
            _, hidden = self.rnn(inputs, hidden)
            
            # Predict next action
            action_logits = self.action_predictor(
                hidden.reshape(batch_size, num_demos, -1)
            )
            
            # Ensure action_logits has correct shape before appending
            if len(action_logits.shape) == 2:  # [batch_size, num_demos, action_dim]
                # Ensure it has action_dim as last dimension
                if action_logits.shape[-1] != self.action_dim:
                    # Reshape if needed
                    action_logits = action_logits.reshape(batch_size, num_demos, self.action_dim)
            
            action_logits_all.append(action_logits)
            
            # Get next action (either from prediction or teacher forcing)
            with torch.no_grad():
                next_actions = action_logits.argmax(dim=-1, keepdim=True)
            
            # Update state embedding if needed (simple approach for test)
            # Normally you'd process the next state through env or dataset
            
            # Add to action sequence
            actions = torch.cat([actions, next_actions], dim=2)
        
        # Stack all logits - resulting in [batch_size, num_demos, seq_length, num_actions]
        if action_logits_all:
            try:
                action_logits_all = torch.stack(action_logits_all, dim=2)
                # Important: Make sure action_logits has shape [batch_size, num_demos, seq_length, num_actions]
                if len(action_logits_all.shape) != 4:
                    print(f"WARNING: Reshaping action_logits from {action_logits_all.shape}")
                    # Handle various cases
                    if len(action_logits_all.shape) == 3:
                        # Could be [batch_size, seq_length, num_actions] missing num_demos dimension
                        if action_logits_all.shape[1] == seq_length and action_logits_all.shape[2] == self.action_dim:
                            action_logits_all = action_logits_all.unsqueeze(1)
                        # Could be [batch_size*num_demos, seq_length, num_actions]
                        else:
                            action_logits_all = action_logits_all.reshape(batch_size, num_demos, seq_length, self.action_dim)
            except Exception as e:
                print(f"Error stacking action_logits: {e}")
                # Provide empty tensor with correct shape 
                action_logits_all = torch.zeros(batch_size, num_demos, seq_length, self.action_dim, device=s_h.device)
        else:
            # Provide empty tensor with correct shape if no predictions
            action_logits_all = torch.zeros(batch_size, num_demos, 0, self.action_dim, device=s_h.device)
            
        print(f"ConditionPolicy - final action_logits_all shape: {action_logits_all.shape}")
        print(f"ConditionPolicy - action_masks shape: {action_masks.shape}")
        
        return None, None, None, action_logits_all, action_masks, None

# Dataset for our simplified test
class KarelDataset(Dataset):
    def __init__(self, dataset_path, id_path, num_agent_actions, max_demo_length=50, device='cpu'):
        self.device = device
        self.max_demo_length = max_demo_length
        self.num_agent_actions = num_agent_actions
        
        # Load data
        self.hdf5_file = h5py.File(dataset_path, 'r')
        with open(id_path, 'r') as f:
            self.id_list = [line.strip().split()[0] for line in f.readlines()]
        
        # Filter IDs that exist in the dataset
        self.valid_ids = []
        for program_id in self.id_list:
            if program_id in self.hdf5_file:
                self.valid_ids.append(program_id)
        
        print(f"Found {len(self.valid_ids)} valid programs in the dataset")

    def __len__(self):
        return len(self.valid_ids)
    
    def __getitem__(self, idx):
        program_id = self.valid_ids[idx]
        
        # Get program data
        program = np.copy(self.hdf5_file[program_id]['program'][()])
        
        # Get execution data
        s_h, a_h, a_h_len = self.get_exec_data(program_id)
        
        # Convert to tensors
        s_h = torch.tensor(s_h, dtype=torch.float32, device=self.device)
        a_h = torch.tensor(a_h, dtype=torch.long, device=self.device)
        a_h_len = torch.tensor(a_h_len, dtype=torch.long, device=self.device)
        
        # Check dimensions and standardize
        # Make sure s_h is always 1 step longer than a_h (for state transitions)
        if s_h.shape[1] != a_h.shape[1] + 1:
            if s_h.shape[1] > a_h.shape[1] + 1:
                # Trim extra states
                s_h = s_h[:, :a_h.shape[1] + 1]
            else:
                # Pad s_h with zeros to be one step longer than a_h
                pad_len = (a_h.shape[1] + 1) - s_h.shape[1]
                s_h_pad = torch.zeros((s_h.shape[0], pad_len) + s_h.shape[2:], device=self.device)
                s_h = torch.cat([s_h, s_h_pad], dim=1)
        
        # For training, we'll use s_h[:,:a_h.shape[1]] as inputs and predict a_h
        # This ensures exact alignment in the network
                
        return program_id, s_h, a_h, a_h_len
    
    def get_exec_data(self, program_id):
        # Process execution data similar to the original codebase
        s_h = np.moveaxis(np.copy(self.hdf5_file[program_id]['s_h']), [-1,-2,-3], [-3,-1,-2])
        a_h = np.copy(self.hdf5_file[program_id]['a_h'])
        s_h_len = np.copy(self.hdf5_file[program_id]['s_h_len'])
        a_h_len = np.copy(self.hdf5_file[program_id]['a_h_len'])
        
        # Print actual shapes from the dataset for debugging
        print(f"Raw s_h shape: {s_h.shape}")
        print(f"Raw a_h shape: {a_h.shape}")
        print(f"Raw s_h_len shape: {s_h_len.shape}")
        print(f"Raw a_h_len shape: {a_h_len.shape}")
        
        # Set a standard target sequence length for consistency
        target_a_h_len = self.max_demo_length - 1  # -1 to leave room for initial state
        
        # Expand demo length if max_demo_len==1
        if s_h.shape[1] == 1:
            s_h = np.concatenate((np.copy(s_h), np.copy(s_h)), axis=1)
            a_h = np.ones((s_h.shape[0], 1), dtype=np.int64)
        
        # Add no-op actions for empty demonstrations
        for i in range(s_h_len.shape[0]):
            if a_h_len[i] == 0:
                a_h_len[i] += 1
                s_h_len[i] += 1
                if s_h[i].shape[0] > 1:  # Check if we have multiple timesteps
                    s_h[i][1, :, :, :] = s_h[i][0, :, :, :]
                a_h[i][0] = self.num_agent_actions - 1
        
        # Set a standard size for a_h
        if a_h.shape[1] > target_a_h_len:
            # Trim a_h if too long
            a_h = a_h[:, :target_a_h_len]
            a_h_len = np.minimum(a_h_len, target_a_h_len)
        else:
            # Pad a_h if too short
            pad_len = target_a_h_len - a_h.shape[1]
            a_h_pad = np.full((a_h.shape[0], pad_len), self.num_agent_actions - 1, dtype=np.int64)
            a_h = np.concatenate([a_h, a_h_pad], axis=1)
        
        # Now ensure s_h is exactly one step longer than a_h
        target_s_h_len = a_h.shape[1] + 1
        
        if s_h.shape[1] < target_s_h_len:
            # Need to add states
            pad_len = target_s_h_len - s_h.shape[1]
            # Copy the last state as padding
            last_state = s_h[:, -1:].repeat(pad_len, axis=1)
            s_h = np.concatenate([s_h, last_state], axis=1)
        elif s_h.shape[1] > target_s_h_len:
            # Need to remove states
            s_h = s_h[:, :target_s_h_len]
        
        # Print final shapes
        print(f"Final s_h shape: {s_h.shape}")
        print(f"Final a_h shape: {a_h.shape}")
        print(f"Final a_h_len shape: {a_h_len.shape}")
        
        # Final verification
        assert s_h.shape[1] == a_h.shape[1] + 1, f"s_h shape: {s_h.shape}, a_h shape: {a_h.shape}"
        
        return s_h, a_h, a_h_len

# The loss calculation
def calculate_behavior_condition_loss(a_h, a_h_len, action_logits, action_masks, num_actions):
    """
    Calculate behavior condition loss (cross entropy loss for predicting next actions in sequences)
    
    Args:
        a_h: Action histories [batch_size, num_demos, seq_len]
        a_h_len: Action history lengths [batch_size, num_demos]
        action_logits: Predicted action logits 
                      Shape should be [batch_size, num_demos, seq_len-1, num_actions]
        action_masks: Valid action masks [batch_size, num_demos, seq_len, 1]
        num_actions: Number of possible actions
        
    Returns:
        loss: Cross entropy loss for next action prediction
        accuracy: Accuracy of next action prediction
    """
    try:
        # Debug prints
        print(f"calculate_behavior_condition_loss - a_h shape: {a_h.shape}")
        print(f"calculate_behavior_condition_loss - a_h_len shape: {a_h_len.shape}")
        print(f"calculate_behavior_condition_loss - action_logits shape: {action_logits.shape}")
        if action_masks is not None:
            print(f"calculate_behavior_condition_loss - action_masks shape: {action_masks.shape}")
        
        # Get shapes
        batch_size, num_demos, max_seq_len = a_h.shape
        
        # Ensure action_logits has the expected 4D shape [batch_size, num_demos, seq_len-1, num_actions]
        if len(action_logits.shape) == 3:
            # If shape is [batch_size*num_demos, seq_len-1, num_actions]
            seq_len_minus_1, action_dim = action_logits.shape[1], action_logits.shape[2]
            action_logits = action_logits.reshape(batch_size, num_demos, seq_len_minus_1, action_dim)
            print(f"calculate_behavior_condition_loss - reshaping action_logits to {action_logits.shape}")
            
        # Ensure action_logits has one less time step than a_h
        # This is because we predict the next action at each step
        if action_logits.shape[2] != max_seq_len - 1:
            if action_logits.shape[2] < max_seq_len - 1:
                # Pad action_logits if it's shorter than expected
                padding = torch.zeros(
                    batch_size, num_demos, (max_seq_len-1) - action_logits.shape[2], action_logits.shape[3],
                    device=action_logits.device
                )
                action_logits = torch.cat([action_logits, padding], dim=2)
                print(f"calculate_behavior_condition_loss - padded action_logits to {action_logits.shape}")
            else:
                # Trim action_logits if it's longer than expected
                action_logits = action_logits[:, :, :max_seq_len-1]
                print(f"calculate_behavior_condition_loss - trimmed action_logits to {action_logits.shape}")
        
        # We predict next actions, so targets are actions shifted by 1
        targets = a_h[:, :, 1:]  # Shape: [batch_size, num_demos, max_seq_len-1]
        print(f"calculate_behavior_condition_loss - targets shape: {targets.shape}")
        
        # Create mask for valid actions (not padding)
        # We use num_actions-1 as the padding token
        valid_actions_mask = (targets != num_actions - 1).float()  # [batch_size, num_demos, max_seq_len-1]
        print(f"calculate_behavior_condition_loss - valid_actions_mask shape: {valid_actions_mask.shape}")
        
        # Combine with action_masks if provided
        if action_masks is not None:
            # Ensure action_masks is the right shape
            if action_masks.shape[2] != max_seq_len:
                print(f"Warning: action_masks has {action_masks.shape[2]} steps but a_h has {max_seq_len}")
                if action_masks.shape[2] > max_seq_len:
                    # Trim if larger
                    action_masks = action_masks[:, :, :max_seq_len]
                else:
                    # Pad if smaller
                    pad_size = max_seq_len - action_masks.shape[2]
                    action_masks_padding = torch.zeros(
                        batch_size, num_demos, pad_size, action_masks.shape[3],
                        device=action_masks.device
                    )
                    action_masks = torch.cat([action_masks, action_masks_padding], dim=2)
            
            # Shift action_masks to match targets
            action_masks = action_masks[:, :, 1:, :]  # [batch_size, num_demos, max_seq_len-1, 1]
            action_masks = action_masks.squeeze(-1)   # [batch_size, num_demos, max_seq_len-1]
            
            # Combine masks
            combined_mask = valid_actions_mask * action_masks
        else:
            combined_mask = valid_actions_mask
        
        # Flatten for loss calculation
        flat_logits = action_logits.reshape(-1, action_logits.shape[-1])  # [batch_size*num_demos*seq_len, num_actions]
        flat_targets = targets.reshape(-1)  # [batch_size*num_demos*seq_len]
        flat_mask = combined_mask.reshape(-1)  # [batch_size*num_demos*seq_len]
        
        # Get indices of valid elements (non-zero in mask)
        valid_idx = flat_mask.nonzero().squeeze(-1)
        
        # If no valid actions, return dummy values
        if valid_idx.numel() == 0:
            print("Warning: No valid actions found for loss calculation")
            return torch.tensor(0.0, device=action_logits.device), torch.tensor(0.0, device=action_logits.device)
        
        # Extract masked elements
        masked_logits = flat_logits[valid_idx]
        masked_targets = flat_targets[valid_idx]
        
        # Compute cross entropy loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(masked_logits, masked_targets.long())
        
        # Compute accuracy
        _, predicted = masked_logits.max(1)
        accuracy = (predicted == masked_targets).float().mean()
        
        return loss, accuracy
    
    except Exception as e:
        print(f"Error in calculate_behavior_condition_loss: {e}")
        import traceback
        traceback.print_exc()
        # Return dummy values to continue training
        return torch.tensor(0.0, device=a_h.device, requires_grad=True), torch.tensor(0.0, device=a_h.device)

def main():
    parser = argparse.ArgumentParser(description='Behavior Condition Loss Test')
    parser.add_argument('--dataset_path', type=str, default='/Users/vincent_tiono/Desktop/HPRL/karel_dataset/data.hdf5')
    parser.add_argument('--id_path', type=str, default='/Users/vincent_tiono/Desktop/HPRL/karel_dataset/id.txt')
    parser.add_argument('--batch_size', type=int, default=2)  # Smaller batch size for debugging
    parser.add_argument('--num_epochs', type=int, default=2)  # Fewer epochs for debugging
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_demo_length', type=int, default=10)  # Shorter demos for debugging
    parser.add_argument('--latent_dim', type=int, default=32)  # Smaller latent dimension for faster processing
    parser.add_argument('--hidden_dim', type=int, default=32)  # Smaller hidden dimension for faster processing
    parser.add_argument('--debug', action='store_true', default=True, help='Run in debug mode with limited data')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--sample_limit', type=int, default=5, help='Limit dataset to this many samples when debugging')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    # Print device info
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    try:
        # Load DSL to get number of actions
        dsl = get_DSL_option_v2(seed=args.seed, environment='karel')
        num_agent_actions = len(dsl.action_functions) + 1  # +1 for no-op
        print(f"Number of agent actions: {num_agent_actions}")
        
        # Create dataset
        dataset = KarelDataset(
            args.dataset_path, 
            args.id_path,
            num_agent_actions,
            max_demo_length=args.max_demo_length,
            device=device
        )
        
        # Limit dataset size for debugging
        if args.debug:
            print(f"Running in debug mode with limited data: {args.sample_limit} samples")
            sample_limit = min(len(dataset.valid_ids), args.sample_limit)
            dataset.valid_ids = dataset.valid_ids[:sample_limit]
            print(f"Dataset size reduced to {len(dataset.valid_ids)} samples")
        
        if len(dataset.valid_ids) == 0:
            print("ERROR: No valid samples in dataset. Please check dataset path and ID file.")
            return
            
        # Create dataloader
        dataloader = DataLoader(
            dataset, 
            batch_size=min(args.batch_size, len(dataset.valid_ids)),  # Ensure batch size isn't larger than dataset
            shuffle=True,
            collate_fn=lambda batch: (
                [x[0] for x in batch],  # program_ids
                torch.stack([x[1] for x in batch]),  # s_h
                torch.stack([x[2] for x in batch]),  # a_h
                torch.stack([x[3] for x in batch])   # a_h_len
            )
        )
        
        if len(dataloader) == 0:
            print("ERROR: Empty dataloader. Please check dataset path and ID file.")
            return
            
        # Check first batch to verify shapes
        print("Checking first batch shapes...")
        try:
            first_batch = next(iter(dataloader))
            program_ids, s_h, a_h, a_h_len = first_batch
            print(f"First batch - s_h shape: {s_h.shape}")
            print(f"First batch - a_h shape: {a_h.shape}")
            print(f"First batch - a_h_len shape: {a_h_len.shape}")
            
            # Create models
            state_shape = (s_h.shape[3], s_h.shape[4], s_h.shape[5])  # Dynamically get state shape
            print(f"Using state shape: {state_shape}")
        except Exception as e:
            print(f"Error checking first batch: {e}")
            import traceback
            traceback.print_exc()
            return
        
        behavior_encoder = SimpleBehaviorEncoder(
            state_shape=state_shape,
            action_dim=num_agent_actions,
            hidden_dim=args.hidden_dim,
            latent_dim=args.latent_dim
        ).to(device)
        
        condition_policy = SimpleConditionPolicy(
            state_shape=state_shape,
            action_dim=num_agent_actions,
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim
        ).to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(
            list(behavior_encoder.parameters()) + list(condition_policy.parameters()),
            lr=args.learning_rate
        )
        
        # Training loop
        for epoch in range(args.num_epochs):
            total_loss = 0
            total_accuracy = 0
            batch_count = 0
            
            print(f"Epoch {epoch+1}/{args.num_epochs}")
            for batch_idx, (program_ids, s_h, a_h, a_h_len) in enumerate(tqdm(dataloader)):
                try:
                    # Reset gradients
                    optimizer.zero_grad()
                    
                    # Print batch shapes for debugging
                    if batch_idx == 0:
                        print(f"Batch {batch_idx} - s_h shape: {s_h.shape}")
                        print(f"Batch {batch_idx} - a_h shape: {a_h.shape}")
                        print(f"Batch {batch_idx} - a_h_len shape: {a_h_len.shape}")
                    
                    # Get behavior embeddings from behavior encoder
                    b_z, b_mean, b_logvar = behavior_encoder(s_h, a_h, a_h_len)
                    
                    # Get action predictions from condition policy
                    _, _, _, action_logits, action_masks, _ = condition_policy(s_h, a_h, b_z)
                    
                    # Calculate behavior condition loss
                    loss, accuracy = calculate_behavior_condition_loss(
                        a_h, a_h_len, action_logits, action_masks, num_agent_actions
                    )
                    
                    # Add KL divergence loss for VAE regularization
                    kl_loss = -0.5 * torch.sum(1 + b_logvar - b_mean.pow(2) - b_logvar.exp())
                    kl_loss = kl_loss / (batch_size * num_demos)  # Normalize by batch size
                    
                    # Total loss
                    total_loss_batch = loss + 0.1 * kl_loss
                    
                    # Backpropagation
                    total_loss_batch.backward()
                    torch.nn.utils.clip_grad_norm_(list(behavior_encoder.parameters()) + list(condition_policy.parameters()), 1.0)
                    optimizer.step()
                    
                    # Track metrics
                    total_loss += loss.item()
                    total_accuracy += accuracy.item()
                    batch_count += 1
                    
                    # Print batch progress
                    if (batch_idx + 1) % 2 == 0:
                        print(f"  Batch {batch_idx+1}/{len(dataloader)}: Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
                        
                    # Break after a few batches if in debug mode
                    if args.debug and batch_idx >= 1:
                        print("Debug mode: breaking after 2 batches")
                        break
                
                except Exception as e:
                    print(f"Error in batch {batch_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Print epoch statistics
            if batch_count > 0:
                avg_loss = total_loss / batch_count
                avg_accuracy = total_accuracy / batch_count
                print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}, Avg Accuracy: {avg_accuracy:.4f}")
        
        print("Training complete!")
        
        # Test the model on a few examples
        print("\nTesting behavior condition on a few examples:")
        with torch.no_grad():
            for i, (program_ids, s_h, a_h, a_h_len) in enumerate(dataloader):
                if i >= 1:  # Just test on 1 batch
                    break
                    
                # Get behavior embeddings
                b_z, _, _ = behavior_encoder(s_h, a_h, a_h_len)
                
                # Get action predictions
                _, _, _, action_logits, action_masks, _ = condition_policy(s_h, a_h, b_z)
                
                # Calculate loss and accuracy
                loss, accuracy = calculate_behavior_condition_loss(
                    a_h, a_h_len, action_logits, action_masks, num_agent_actions
                )
                
                print(f"Test Batch {i+1} - Loss: {loss.item():.4f}, Accuracy: {accuracy.item():.4f}")
                
                # Show predictions vs actual for first sequence in batch
                if len(action_logits.shape) == 4:
                    preds = action_logits[0, 0].argmax(dim=-1)
                else:
                    preds = action_logits[0].argmax(dim=-1)
                    
                targets = a_h[0, 0, 1:]  # Skip first token
                
                # Get valid actions (not padding)
                valid_idx = (targets != num_agent_actions - 1).nonzero().squeeze()
                if valid_idx.numel() > 0:
                    if valid_idx.dim() == 0:  # handle scalar case
                        valid_idx = valid_idx.unsqueeze(0)
                        
                    valid_preds = preds[valid_idx]
                    valid_targets = targets[valid_idx]
                    
                    print("  Example predictions:")
                    print(f"  Predicted: {valid_preds.cpu().numpy()}")
                    print(f"  Actual:    {valid_targets.cpu().numpy()}")
                    print(f"  Matches:   {(valid_preds == valid_targets).cpu().numpy()}")
    
    except Exception as e:
        print(f"Error in script execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 