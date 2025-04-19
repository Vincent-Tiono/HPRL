"""
Behavior Encoder Analysis Tool

This script performs detailed analysis of the behavior encoder to diagnose
why the encoded vectors are clustered in a corner of the PCA plot.

Author: Vincent Chang
"""

import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
from pretrain.models_option_new_vae import ActionBehaviorEncoder, ProgramEncoder
from torch.nn import functional as F
import torch.nn as nn
from dataclasses import dataclass
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


@dataclass
class EncoderConfig:
    """Configuration for encoder models."""
    recurrent_policy: bool = True
    num_agent_actions: int = 6
    num_lstm_cell_units: int = 64
    rnn_type: str = 'GRU'
    dropout: float = 0.0
    use_linear: bool = True
    num_rnn_encoder_units: int = 256
    input_channel: int = 8
    input_height: int = 8
    input_width: int = 8
    fuse_s_0: bool = False


def load_encoder(model_path='final_params.ptp'):
    """Load behavior encoder from saved weights."""
    params_list = torch.load(model_path, map_location=torch.device('cpu'))
    param_dict = params_list[0]
    
    config = EncoderConfig()
    behavior_encoder = ActionBehaviorEncoder(
        recurrent=config.recurrent_policy,
        num_actions=config.num_agent_actions,
        hidden_size=config.num_lstm_cell_units,
        rnn_type=config.rnn_type,
        dropout=config.dropout,
        use_linear=config.use_linear,
        unit_size=config.num_rnn_encoder_units,
        input_channel=config.input_channel,
        input_height=config.input_height,
        input_width=config.input_width,
        fuse_s_0=config.fuse_s_0
    )
    
    state_dict = {
        key.replace('vae.behavior_encoder.', ''): value
        for key, value in param_dict.items()
        if key.startswith('vae.behavior_encoder')
    }
    
    behavior_encoder.load_state_dict(state_dict)
    behavior_encoder.eval()
    return behavior_encoder


def process_hdf5_file(file_path, num_agent_actions=6):
    """Process HDF5 file containing program execution data."""
    try:
        with h5py.File(file_path, 'r') as f:
            programs = []
            sample_keys = [k for k in f.keys() if not k.startswith('data_info_')]
            
            for program_id in sample_keys[:100]:  # Limit to 100 programs for quicker analysis
                try:
                    if program_id not in f or 's_h' not in f[program_id]:
                        continue
                        
                    s_h = np.moveaxis(np.copy(f[program_id]['s_h']), [-1,-2,-3], [-3,-1,-2])
                    a_h = np.copy(f[program_id]['a_h'])
                    s_h_len = np.copy(f[program_id]['s_h_len'])
                    a_h_len = np.copy(f[program_id]['a_h_len'])
                    
                    if s_h.shape[1] == 1:
                        s_h = np.concatenate((np.copy(s_h), np.copy(s_h)), axis=1)
                        a_h = np.ones((s_h.shape[0], 1))
                        
                    for i in range(s_h_len.shape[0]):
                        if a_h_len[i] == 0:
                            assert s_h_len[i] == 1
                            a_h_len[i] += 1
                            s_h_len[i] += 1
                            s_h[i][1] = s_h[i][0]
                            a_h[i][0] = num_agent_actions - 1
                    
                    programs.append((program_id, s_h, s_h_len, a_h, a_h_len))
                except Exception as e:
                    print(f"Error processing program {program_id}: {e}")
            return programs
    except Exception as e:
        print(f"Error opening file {file_path}: {e}")
        return []


def register_hooks(model):
    """Register hooks to capture intermediate activations."""
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    # Register hooks for GRU/LSTM, action encoder, state encoder, and any projections
    model.gru.register_forward_hook(get_activation('gru'))
    model.action_encoder.register_forward_hook(get_activation('action_encoder'))
    model.state_encoder.register_forward_hook(get_activation('state_encoder'))
    model.state_projector.register_forward_hook(get_activation('state_projector'))
    if hasattr(model, 'proj'):
        model.proj.register_forward_hook(get_activation('proj'))
    
    return activation


def analyze_encoder_weights(encoder):
    """Analyze encoder weights and structure."""
    print("===== Behavior Encoder Structure & Weight Analysis =====")
    
    # Analyze action embedding weights
    action_emb_weights = encoder.action_encoder.weight.data
    print(f"Action Embedding: shape={action_emb_weights.shape}")
    print(f"  Mean: {action_emb_weights.mean().item():.6f}")
    print(f"  Std Dev: {action_emb_weights.std().item():.6f}")
    print(f"  Min: {action_emb_weights.min().item():.6f}")
    print(f"  Max: {action_emb_weights.max().item():.6f}")
    
    # Analyze state encoder weights
    print("\nState Encoder:")
    for name, module in encoder.state_encoder.named_modules():
        if hasattr(module, 'weight'):
            w = module.weight.data
            print(f"  {name}: shape={w.shape}")
            print(f"    Mean: {w.mean().item():.6f}")
            print(f"    Std Dev: {w.std().item():.6f}")
            print(f"    Min: {w.min().item():.6f}")
            print(f"    Max: {w.max().item():.6f}")
    
    # Analyze state projector weights
    w = encoder.state_projector.weight.data
    print(f"\nState Projector: shape={w.shape}")
    print(f"  Mean: {w.mean().item():.6f}")
    print(f"  Std Dev: {w.std().item():.6f}")
    print(f"  Min: {w.min().item():.6f}")
    print(f"  Max: {w.max().item():.6f}")
    
    # Analyze GRU weights
    w_ih = encoder.gru.weight_ih_l0.data
    w_hh = encoder.gru.weight_hh_l0.data
    
    print(f"\nGRU Input->Hidden weights: shape={w_ih.shape}")
    print(f"  Mean: {w_ih.mean().item():.6f}")
    print(f"  Std Dev: {w_ih.std().item():.6f}")
    print(f"  Min: {w_ih.min().item():.6f}")
    print(f"  Max: {w_ih.max().item():.6f}")
    
    print(f"\nGRU Hidden->Hidden weights: shape={w_hh.shape}")
    print(f"  Mean: {w_hh.mean().item():.6f}")
    print(f"  Std Dev: {w_hh.std().item():.6f}")
    print(f"  Min: {w_hh.min().item():.6f}")
    print(f"  Max: {w_hh.max().item():.6f}")
    
    # Check for projection layer
    if hasattr(encoder, 'proj'):
        w = encoder.proj.weight.data
        print(f"\nProjection Layer: shape={w.shape}")
        print(f"  Mean: {w.mean().item():.6f}")
        print(f"  Std Dev: {w.std().item():.6f}")
        print(f"  Min: {w.min().item():.6f}")
        print(f"  Max: {w.max().item():.6f}")


def analyze_activations(encoder, program):
    """Analyze activations when encoding a program."""
    program_id, s_h, s_h_len, a_h, a_h_len = program
    
    # Convert to torch tensors
    s_h_tensor = torch.tensor(s_h, dtype=torch.float32)  # (num_demos, T, C, H, W)
    s_h_len_tensor = torch.tensor(s_h_len, dtype=torch.int16)  # (num_demos)
    a_h_tensor = torch.tensor(a_h, dtype=torch.int16)   # (num_demos, T)
    a_h_len_tensor = torch.tensor(a_h_len, dtype=torch.int16)  # (num_demos)
    
    # Add batch dimension (B=1)
    s_h_batch = s_h_tensor.unsqueeze(0)  # (1, num_demos, T, C, H, W)
    a_h_batch = a_h_tensor.unsqueeze(0)  # (1, num_demos, T)
    s_h_len_batch = s_h_len_tensor.unsqueeze(0)  # (1, num_demos)
    a_h_len_batch = a_h_len_tensor.unsqueeze(0)  # (1, num_demos)
    
    # Register hooks
    activation = register_hooks(encoder)
    
    # Forward pass with hooks
    with torch.no_grad():
        output = encoder(s_h_batch, a_h_batch, s_h_len_batch, a_h_len_batch)
    
    # Analyze intermediate activations
    print("\n===== Activation Analysis =====")
    print(f"Program ID: {program_id}")
    
    # State encoder output
    if 'state_encoder' in activation:
        act = activation['state_encoder']
        print(f"\nState Encoder Output: shape={act.shape}")
        print(f"  Mean: {act.mean().item():.6f}")
        print(f"  Std Dev: {act.std().item():.6f}")
        print(f"  Min: {act.min().item():.6f}")
        print(f"  Max: {act.max().item():.6f}")
        print(f"  Near Zero: {torch.sum(torch.isclose(act, torch.zeros_like(act), atol=1e-6)).item()} out of {act.numel()}")
    
    # Action encoder output
    if 'action_encoder' in activation:
        act = activation['action_encoder']
        print(f"\nAction Encoder Output: shape={act.shape}")
        print(f"  Mean: {act.mean().item():.6f}")
        print(f"  Std Dev: {act.std().item():.6f}")
        print(f"  Min: {act.min().item():.6f}")
        print(f"  Max: {act.max().item():.6f}")
        print(f"  Near Zero: {torch.sum(torch.isclose(act, torch.zeros_like(act), atol=1e-6)).item()} out of {act.numel()}")
    
    # GRU output
    if 'gru' in activation:
        act = activation['gru']
        print(f"\nGRU Output: shape={act.shape}")
        print(f"  Mean: {act.mean().item():.6f}")
        print(f"  Std Dev: {act.std().item():.6f}")
        print(f"  Min: {act.min().item():.6f}")
        print(f"  Max: {act.max().item():.6f}")
        print(f"  Near Zero: {torch.sum(torch.isclose(act, torch.zeros_like(act), atol=1e-6)).item()} out of {act.numel()}")
    
    # Final output
    print(f"\nFinal Output: shape={output.shape}")
    print(f"  Mean: {output.mean().item():.6f}")
    print(f"  Std Dev: {output.std().item():.6f}")
    print(f"  Min: {output.min().item():.6f}")
    print(f"  Max: {output.max().item():.6f}")
    print(f"  Near Zero: {torch.sum(torch.isclose(output, torch.zeros_like(output), atol=1e-6)).item()} out of {output.numel()}")
    
    # Check for saturation (how many values are close to max/min)
    near_max = torch.sum(torch.isclose(output, torch.ones_like(output), atol=1e-2)).item()
    near_min = torch.sum(torch.isclose(output, -torch.ones_like(output), atol=1e-2)).item()
    print(f"  Near Max (1.0): {near_max} out of {output.numel()} ({near_max/output.numel():.2%})")
    print(f"  Near Min (-1.0): {near_min} out of {output.numel()} ({near_min/output.numel():.2%})")
    
    return output, activation


def encode_multiple_programs(encoder, programs, max_programs=100):
    """Encode multiple programs and analyze the distribution of vectors."""
    encoded_vectors = []
    activations = []
    program_ids = []
    
    print(f"\n===== Analyzing {min(len(programs), max_programs)} Programs =====")
    
    for i, program in enumerate(programs[:max_programs]):
        program_id = program[0]
        try:
            with torch.no_grad():
                output, activation = analyze_activations(encoder, program)
                
            encoded_vectors.append(output.cpu().numpy())
            activations.append(activation)
            program_ids.append(program_id)
            
            if i % 10 == 0:
                print(f"Processed {i+1} programs...")
        except Exception as e:
            print(f"Error processing program {program_id}: {e}")
            traceback.print_exc()
    
    # Convert to numpy array for analysis
    encoded_array = np.vstack(encoded_vectors)
    
    # Perform statistical analysis
    print("\n===== Vector Distribution Analysis =====")
    print(f"Shape: {encoded_array.shape}")
    print(f"Mean: {np.mean(encoded_array):.6f}")
    print(f"Std Dev: {np.std(encoded_array):.6f}")
    print(f"Min: {np.min(encoded_array):.6f}")
    print(f"Max: {np.max(encoded_array):.6f}")
    
    # Calculate per-dimension statistics
    dim_means = np.mean(encoded_array, axis=0)
    dim_stds = np.std(encoded_array, axis=0)
    dim_mins = np.min(encoded_array, axis=0)
    dim_maxs = np.max(encoded_array, axis=0)
    
    print("\nPer-dimension statistics:")
    print(f"  Mean range: {np.min(dim_means):.6f} to {np.max(dim_means):.6f}")
    print(f"  Std Dev range: {np.min(dim_stds):.6f} to {np.max(dim_stds):.6f}")
    
    # Check for inactive dimensions (very low variance)
    inactive_dims = np.sum(dim_stds < 0.01)
    print(f"  Dimensions with very low variation (std < 0.01): {inactive_dims} out of {dim_stds.size} ({inactive_dims/dim_stds.size:.2%})")
    
    # Check for saturated dimensions (values clustered at extremes)
    saturated_min = np.sum(dim_mins > -0.1)
    saturated_max = np.sum(dim_maxs < 0.1)
    print(f"  Dimensions with min > -0.1: {saturated_min} out of {dim_mins.size} ({saturated_min/dim_mins.size:.2%})")
    print(f"  Dimensions with max < 0.1: {saturated_max} out of {dim_maxs.size} ({saturated_max/dim_maxs.size:.2%})")
    
    return encoded_array, program_ids, activations


def visualize_vectors(vectors, title="Behavior Encoder Output Distribution"):
    """Create visualizations for the encoded vectors."""
    # Create a directory for visualizations
    save_dir = Path("behavior_encoder_visualizations")
    save_dir.mkdir(exist_ok=True)
    
    # 1. Histogram of vector norms
    norms = np.linalg.norm(vectors, axis=1)
    plt.figure(figsize=(12, 6))
    plt.hist(norms, bins=50)
    plt.title("Distribution of Vector Norms")
    plt.xlabel("Vector Norm")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "vector_norms_histogram.png")
    plt.close()
    
    # 2. Vector component heatmap (sample up to 100 vectors)
    max_vectors = min(100, vectors.shape[0])
    plt.figure(figsize=(15, 10))
    sns.heatmap(vectors[:max_vectors], cmap='viridis')
    plt.title("Vector Components Heatmap")
    plt.xlabel("Dimension")
    plt.ylabel("Vector Index")
    plt.savefig(save_dir / "vector_components_heatmap.png")
    plt.close()
    
    # 3. Pairwise cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    max_vectors = min(100, vectors.shape[0])
    similarities = cosine_similarity(vectors[:max_vectors])
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarities, cmap='coolwarm')
    plt.title("Pairwise Cosine Similarity")
    plt.savefig(save_dir / "cosine_similarity_heatmap.png")
    plt.close()
    
    # 4. PCA visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(vectors)
    
    # 2D PCA plot
    plt.figure(figsize=(12, 10))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.7)
    plt.title(f"PCA Projection of {title}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "pca_2d.png")
    plt.close()
    
    # 3D PCA plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], alpha=0.7)
    ax.set_title(f"3D PCA Projection of {title}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.2%} variance)")
    plt.savefig(save_dir / "pca_3d.png")
    plt.close()
    
    # 5. Dimension-wise statistics
    dim_means = np.mean(vectors, axis=0)
    dim_stds = np.std(vectors, axis=0)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(len(dim_means)), dim_means)
    plt.title("Mean Value by Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Mean Value")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.bar(range(len(dim_stds)), dim_stds)
    plt.title("Standard Deviation by Dimension")
    plt.xlabel("Dimension")
    plt.ylabel("Standard Deviation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "dimension_statistics.png")
    plt.close()
    

def investigate_collapsed_dims(vectors):
    """Investigate if there are collapsed dimensions causing the clustering."""
    # Create a directory for visualizations
    save_dir = Path("behavior_encoder_visualizations")
    save_dir.mkdir(exist_ok=True)
    
    # Check for dimensions with low variance
    dim_vars = np.var(vectors, axis=0)
    dim_means = np.mean(vectors, axis=0)
    
    # Sort dimensions by variance
    sorted_indices = np.argsort(dim_vars)
    low_var_indices = sorted_indices[:10]  # 10 lowest variance dimensions
    high_var_indices = sorted_indices[-10:]  # 10 highest variance dimensions
    
    print("\n===== Dimension Variance Analysis =====")
    print("Dimensions with lowest variance:")
    for i, idx in enumerate(low_var_indices):
        print(f"  Dim {idx}: var = {dim_vars[idx]:.6f}, mean = {dim_means[idx]:.6f}")
    
    print("\nDimensions with highest variance:")
    for i, idx in enumerate(high_var_indices):
        print(f"  Dim {idx}: var = {dim_vars[idx]:.6f}, mean = {dim_means[idx]:.6f}")
    
    # Plot distribution of values for low-variance dimensions
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(low_var_indices[:5]):
        plt.subplot(5, 1, i+1)
        plt.hist(vectors[:, idx], bins=50, alpha=0.7)
        plt.title(f"Dimension {idx} (var={dim_vars[idx]:.6f})")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "low_variance_dims_distribution.png")
    plt.close()
    
    # Plot distribution of values for high-variance dimensions
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(high_var_indices[-5:]):
        plt.subplot(5, 1, i+1)
        plt.hist(vectors[:, idx], bins=50, alpha=0.7)
        plt.title(f"Dimension {idx} (var={dim_vars[idx]:.6f})")
        plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "high_variance_dims_distribution.png")
    plt.close()
    
    # Compute correlation matrix between dimensions
    corr_matrix = np.corrcoef(vectors, rowvar=False)
    
    # Plot correlation heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Dimension Correlation Matrix")
    plt.savefig(save_dir / "dimension_correlation.png")
    plt.close()
    
    # Check for highly correlated dimensions
    np.fill_diagonal(corr_matrix, 0)  # Exclude self-correlations
    high_corr_pairs = []
    for i in range(corr_matrix.shape[0]):
        for j in range(i+1, corr_matrix.shape[1]):
            if abs(corr_matrix[i, j]) > 0.9:
                high_corr_pairs.append((i, j, corr_matrix[i, j]))
    
    print("\nHighly correlated dimension pairs (|corr| > 0.9):")
    for i, j, corr in high_corr_pairs:
        print(f"  Dims {i} and {j}: corr = {corr:.6f}")
    
    if len(high_corr_pairs) > 0:
        # Plot scatter for some highly correlated pairs
        plt.figure(figsize=(15, 10))
        for plot_idx, (i, j, corr) in enumerate(high_corr_pairs[:min(5, len(high_corr_pairs))]):
            plt.subplot(min(5, len(high_corr_pairs)), 1, plot_idx+1)
            plt.scatter(vectors[:, i], vectors[:, j], alpha=0.5)
            plt.title(f"Dims {i} vs {j} (corr={corr:.6f})")
            plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / "highly_correlated_dims.png")
        plt.close()
    
    return dim_vars, dim_means, low_var_indices, high_var_indices


def compare_original_vectors_with_normalized(vectors):
    """Compare original vectors with L2 normalized vectors to see if normalization helps."""
    # L2 normalize vectors
    normalized_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Create visualizations directory
    save_dir = Path("behavior_encoder_visualizations")
    save_dir.mkdir(exist_ok=True)
    
    # Run PCA on both original and normalized vectors
    from sklearn.decomposition import PCA
    
    pca_orig = PCA(n_components=2)
    pca_norm = PCA(n_components=2)
    
    pca_orig_result = pca_orig.fit_transform(vectors)
    pca_norm_result = pca_norm.fit_transform(normalized_vectors)
    
    # Plot both results side by side
    plt.figure(figsize=(20, 10))
    
    plt.subplot(1, 2, 1)
    plt.scatter(pca_orig_result[:, 0], pca_orig_result[:, 1], alpha=0.7)
    plt.title("Original Vectors PCA")
    plt.xlabel(f"PC1 ({pca_orig.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca_orig.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(pca_norm_result[:, 0], pca_norm_result[:, 1], alpha=0.7)
    plt.title("Normalized Vectors PCA")
    plt.xlabel(f"PC1 ({pca_norm.explained_variance_ratio_[0]:.2%} variance)")
    plt.ylabel(f"PC2 ({pca_norm.explained_variance_ratio_[1]:.2%} variance)")
    plt.grid(alpha=0.3)
    
    plt.savefig(save_dir / "original_vs_normalized_pca.png")
    plt.close()
    
    # Check vector norms
    orig_norms = np.linalg.norm(vectors, axis=1)
    norm_norms = np.linalg.norm(normalized_vectors, axis=1)
    
    plt.figure(figsize=(20, 6))
    plt.subplot(1, 2, 1)
    plt.hist(orig_norms, bins=50, alpha=0.7)
    plt.title("Original Vector Norms")
    plt.xlabel("Norm")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.hist(norm_norms, bins=50, alpha=0.7)
    plt.title("Normalized Vector Norms (should be ~1.0)")
    plt.xlabel("Norm")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    
    plt.savefig(save_dir / "original_vs_normalized_norms.png")
    plt.close()
    
    print("\n===== Original vs Normalized Vectors =====")
    print(f"Original vectors variance explained: {pca_orig.explained_variance_ratio_[0]:.2%}, {pca_orig.explained_variance_ratio_[1]:.2%}")
    print(f"Normalized vectors variance explained: {pca_norm.explained_variance_ratio_[0]:.2%}, {pca_norm.explained_variance_ratio_[1]:.2%}")
    
    return normalized_vectors


def analyze_tanh_behavior(vectors):
    """Analyze if vectors are clustered due to tanh nonlinearity."""
    # Create visualizations directory
    save_dir = Path("behavior_encoder_visualizations")
    save_dir.mkdir(exist_ok=True)
    
    # Flatten vectors for histogram
    flat_values = vectors.flatten()
    
    # Create histogram of all values
    plt.figure(figsize=(12, 6))
    plt.hist(flat_values, bins=100, alpha=0.7)
    plt.axvline(x=-1, color='r', linestyle='--', label='tanh min (-1)')
    plt.axvline(x=1, color='g', linestyle='--', label='tanh max (1)')
    plt.title("Distribution of Vector Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "tanh_distribution.png")
    plt.close()
    
    # Check for saturation
    near_1 = np.sum((flat_values > 0.9) & (flat_values <= 1.0)) / flat_values.size
    near_minus_1 = np.sum((flat_values < -0.9) & (flat_values >= -1.0)) / flat_values.size
    
    print("\n===== Tanh Saturation Analysis =====")
    print(f"Values near 1 (>0.9): {near_1:.2%}")
    print(f"Values near -1 (<-0.9): {near_minus_1:.2%}")
    print(f"Total saturation: {near_1 + near_minus_1:.2%}")
    
    # Check distribution in tanh brackets
    brackets = [
        (-1.0, -0.9),
        (-0.9, -0.5),
        (-0.5, 0.0),
        (0.0, 0.5),
        (0.5, 0.9),
        (0.9, 1.0)
    ]
    
    percentages = []
    for lower, upper in brackets:
        pct = np.sum((flat_values >= lower) & (flat_values < upper)) / flat_values.size
        percentages.append(pct)
        print(f"Values in [{lower:.1f}, {upper:.1f}): {pct:.2%}")
    
    # Plot distribution by bracket
    plt.figure(figsize=(12, 6))
    plt.bar([f"[{b[0]:.1f}, {b[1]:.1f})" for b in brackets], percentages)
    plt.title("Distribution of Values in tanh Range Brackets")
    plt.ylabel("Percentage of Values")
    plt.grid(alpha=0.3)
    plt.savefig(save_dir / "tanh_bracket_distribution.png")
    plt.close()
    
    return flat_values


def main():
    """Main execution function."""
    try:
        # Load behavior encoder
        behavior_encoder = load_encoder('final_params.ptp')
        
        # Analyze encoder weights
        analyze_encoder_weights(behavior_encoder)
        
        # Process HDF5 file
        hdf5_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/data.hdf5"
        programs = process_hdf5_file(hdf5_file_path, behavior_encoder.num_actions)
        
        # Encode programs and analyze distribution
        encoded_vectors, program_ids, activations = encode_multiple_programs(behavior_encoder, programs)
        
        # Create visualizations
        visualize_vectors(encoded_vectors)
        
        # Investigate collapsed dimensions
        dim_vars, dim_means, low_var_indices, high_var_indices = investigate_collapsed_dims(encoded_vectors)
        
        # Compare original vs normalized vectors
        normalized_vectors = compare_original_vectors_with_normalized(encoded_vectors)
        
        # Analyze tanh behavior
        flat_values = analyze_tanh_behavior(encoded_vectors)
        
        print("\nAnalysis complete. Visualizations saved to the 'behavior_encoder_visualizations' directory.")
        print("Check this directory for plots and insights into the behavior encoder's performance.")
        
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()