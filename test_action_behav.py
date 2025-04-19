"""
PCA Analysis Tool for Program Behavior and Source Code Encodings

This module provides functionality to:
1. Load and manage pre-trained encoders
2. Process program execution data from HDF5 files
3. Process program source code from text files
4. Encode program behaviors and source code
5. Perform PCA analysis and visualization
6. Compare and analyze the encodings

Author: Vincent Chang
Date: 2024
"""

import torch
import h5py
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import traceback
import re
from pretrain.models_option_new_vae import ActionBehaviorEncoder, ProgramEncoder
from karel_env.dsl import get_DSL_option_v2
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass


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
    fuse_s_0: bool = True


class EncoderManager:
    """Manages loading and initialization of behavior and program encoders."""
    
    def __init__(self, model_path: str = 'best_valid_params.ptp'):
        """Initialize the encoder manager.
        
        Args:
            model_path: Path to the saved model parameters
        """
        self.model_path = model_path
        self.behavior_encoder = None
        self.program_encoder = None
        
    def load_encoders(self) -> Tuple[ActionBehaviorEncoder, ProgramEncoder]:
        """Load and initialize the encoders from saved weights.
        
        Returns:
            Tuple containing the behavior encoder and program encoder
        """
        params_list = torch.load(self.model_path, map_location=torch.device('cpu'))
        param_dict = params_list[0]
        
        config = self._create_encoder_config()
        self.behavior_encoder = self._init_behavior_encoder(config, param_dict)
        self.program_encoder = self._init_program_encoder(config, param_dict)
        
        return self.behavior_encoder, self.program_encoder
    
    def _create_encoder_config(self) -> EncoderConfig:
        """Create encoder configuration."""
        return EncoderConfig()
    
    def _init_behavior_encoder(self, config: EncoderConfig, param_dict: Dict) -> ActionBehaviorEncoder:
        """Initialize and load weights for the behavior encoder."""
        encoder = ActionBehaviorEncoder(
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
        
        encoder.load_state_dict(state_dict)
        encoder.eval()
        return encoder
    
    def _init_program_encoder(self, config: EncoderConfig, param_dict: Dict) -> ProgramEncoder:
        """Initialize and load weights for the program encoder."""
        encoder = ProgramEncoder(
            num_inputs=35,  # token space size
            num_outputs=64,  # latent space size
            recurrent=True,
            hidden_size=config.num_lstm_cell_units,
            rnn_type=config.rnn_type,
            dropout=config.dropout,
            use_linear=config.use_linear,
            unit_size=config.num_rnn_encoder_units
        )
        
        state_dict = {
            key.replace('vae.program_encoder.', ''): value
            for key, value in param_dict.items()
            if key.startswith('vae.program_encoder')
        }
        
        encoder.load_state_dict(state_dict)
        encoder.eval()
        return encoder


class DataProcessor:
    """Handles data processing for program execution and source code."""
    
    def __init__(self, dsl):
        """Initialize the data processor.
        
        Args:
            dsl: Domain Specific Language object for parsing programs
        """
        self.dsl = dsl
    
    def process_hdf5_file(self, file_path: str, behavior_encoder: ActionBehaviorEncoder) -> List[Tuple]:
        """Process HDF5 file containing program execution data.
        
        Args:
            file_path: Path to the HDF5 file
            behavior_encoder: Behavior encoder model
            
        Returns:
            List of tuples containing program data
        """
        try:
            with h5py.File(file_path, 'r') as f:
                programs = []
                sample_keys = [k for k in f.keys() if not k.startswith('data_info_')]
                
                for program_id in sample_keys:
                    try:
                        s_h, s_h_len, a_h, a_h_len = self._get_exec_data(f, program_id, behavior_encoder.num_actions)
                        if len(s_h) > 0 and len(a_h) > 0:
                            programs.append((program_id, s_h, s_h_len, a_h, a_h_len))
                    except Exception as e:
                        print(f"Error processing program {program_id}: {e}")
                        traceback.print_exc()
                return programs
        except Exception as e:
            print(f"Error opening file {file_path}: {e}")
            traceback.print_exc()
            return []
    
    def _get_exec_data(self, hdf5_file, program_id: str, num_agent_actions: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract execution data from HDF5 file for a given program."""
        if program_id not in hdf5_file or 's_h' not in hdf5_file[program_id]:
            return np.array([]), np.array([]), np.array([])
            
        s_h = np.moveaxis(np.copy(hdf5_file[program_id]['s_h']), [-1,-2,-3], [-3,-1,-2])
        a_h = np.copy(hdf5_file[program_id]['a_h'])
        s_h_len = np.copy(hdf5_file[program_id]['s_h_len'])
        a_h_len = np.copy(hdf5_file[program_id]['a_h_len'])
        
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
                
        # results = map(lambda x: np.expand_dims(x[0][0], 0), zip(s_h, s_h_len))
        # s_h = np.stack(list(results))
        # print(f"s_h.shape: {s_h.shape}, s_h_len.shape: {s_h_len.shape}, a_h.shape: {a_h.shape}, a_h_len.shape: {a_h_len.shape}\n")

        return s_h, s_h_len, a_h, a_h_len
    
    def load_programs_from_txt(self, file_path: str) -> List[Tuple[str, Optional[str]]]:
        """Load program IDs and code from text file.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of tuples (program_id, program_code)
        """
        program_data = []
        try:
            with open(file_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if parts:
                        program_id = parts[0]
                        program_code = ' '.join(parts[1:]) if len(parts) > 1 else None
                        program_data.append((program_id, program_code))
        except Exception as e:
            print(f"Error loading programs from {file_path}: {e}")
            traceback.print_exc()
        return program_data


class Encoder:
    """Handles encoding of program behaviors and source code."""
    
    @staticmethod
    def encode_demos(programs: List[Tuple], behavior_encoder: ActionBehaviorEncoder) -> Tuple[np.ndarray, List[str]]:
        """Encode program demonstrations using behavior encoder.
        
        Args:
            programs: List of program data tuples
            behavior_encoder: Behavior encoder model
            
        Returns:
            Tuple of encoded vectors and program IDs
        """
        latent_vectors = []
        program_ids = []
        
        for program_id, s_h, s_h_len, a_h, a_h_len in programs:
            try:
                # Convert to torch tensors
                s_h_tensor = torch.tensor(s_h, dtype=torch.float32)  # (num_demos, T, C, H, W)
                s_h_len_tensor = torch.tensor(s_h_len, dtype=torch.int16)  # (num_demos)
                a_h_tensor = torch.tensor(a_h, dtype=torch.int16)   # (num_demos, T)
                a_h_len_tensor = torch.tensor(a_h_len, dtype=torch.int16)  # (num_demos)
                
                # print(f"s_h_tensor.shape: {s_h_tensor.shape}, s_h_len_tensor.shape: {s_h_len_tensor.shape}, a_h_tensor.shape: {a_h_tensor.shape}, a_h_len_tensor.shape: {a_h_len_tensor.shape}\n")

                # BehaviorEncoder expects: 
                # s_h: shape (B, R, T, C, H, W) - B=batch_size, R=num_demos_per_program
                # a_h: shape (B, R, T)
                # s_h_len: shape (B, R)
                # a_h_len: shape (B, R)
                
                # Add batch dimension (B=1)
                s_h_batch = s_h_tensor.unsqueeze(0)  # (1, num_demos, T, C, H, W)
                a_h_batch = a_h_tensor.unsqueeze(0)  # (1, num_demos, T)
                s_h_len_batch = s_h_len_tensor.unsqueeze(0)  # (1, num_demos)
                a_h_len_batch = a_h_len_tensor.unsqueeze(0)  # (1, num_demos)
                
                # print(f"s_h_batch.shape: {s_h_batch.shape}, s_h_len_batch.shape: {s_h_len_batch.shape}, a_h_batch.shape: {a_h_batch.shape}, a_h_len_batch.shape: {a_h_len_batch.shape}\n")
                
                with torch.no_grad():
                    # Forward pass through behavior encoder
                    latent = behavior_encoder(s_h_batch, a_h_batch, s_h_len_batch, a_h_len_batch)
                    # latent shape: (1, out_dim)
                    latent_vectors.append(latent.cpu().numpy())
                    program_ids.append(program_id)
                    
            except Exception as e:
                print(f"Error encoding program {program_id}: {e}")
                traceback.print_exc()
                
        return np.array(latent_vectors), program_ids
    
    @staticmethod
    def encode_programs(program_data: List[Tuple], program_encoder: ProgramEncoder, dsl) -> Tuple[np.ndarray, List[str]]:
        """Encode program source code using program encoder.
        
        Args:
            program_data: List of program data tuples
            program_encoder: Program encoder model
            dsl: DSL object for parsing
            
        Returns:
            Tuple of encoded vectors and program IDs
        """
        latent_vectors = []
        program_ids = []
        
        for program_id, program_code in program_data:
            if not program_code:
                continue
                
            try:
                program_tokens = dsl.str2intseq(program_code)
            except:
                match = re.search(r'DEF run m\(\) \{(.*?)\}', program_code)
                if match:
                    program_code = f"DEF run m() {{{match.group(1)}}}"
                    try:
                        program_tokens = dsl.str2intseq(program_code)
                    except:
                        continue
                else:
                    continue
                    
            tokens_tensor = torch.LongTensor(program_tokens).unsqueeze(0)
            src_len = torch.LongTensor([len(program_tokens)])
            
            with torch.no_grad():
                _, encoder_output = program_encoder(tokens_tensor, src_len)
                z = Encoder._sample_latent(encoder_output.squeeze(), 64)
                latent_vectors.append(z.cpu().numpy())
                program_ids.append(program_id)
                
        return np.array(latent_vectors), program_ids
    
    @staticmethod
    def _sample_latent(h_enc: torch.Tensor, hidden_size: int) -> torch.Tensor:
        """Sample from latent space distribution."""
        _enc_mu = nn.Linear(hidden_size, hidden_size)
        _enc_log_sigma = nn.Linear(hidden_size, hidden_size)
        
        mu = _enc_mu(h_enc)
        log_sigma = _enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float().to(h_enc.device)
        
        return mu + sigma * Variable(std_z, requires_grad=False)


class Analyzer:
    """Performs analysis and visualization of encoded data."""
    
    @staticmethod
    def analyze_encodings(behavior_vectors: np.ndarray, program_vectors: np.ndarray,
                         behavior_ids: List[str], program_ids: List[str]) -> Dict[str, Any]:
        """Analyze and compare encodings.
        
        Args:
            behavior_vectors: Behavior encoding vectors
            program_vectors: Program encoding vectors
            behavior_ids: List of behavior program IDs
            program_ids: List of program IDs
            
        Returns:
            Dictionary containing analysis results
        """
        behavior_ids_set = set(behavior_ids)
        program_ids_set = set(program_ids)
        
        common_ids = behavior_ids_set.intersection(program_ids_set)
        behavior_only_ids = behavior_ids_set - program_ids_set
        program_only_ids = program_ids_set - behavior_ids_set
        
        # Calculate vector differences
        behavior_dict = {pid: vec for pid, vec in zip(behavior_ids, behavior_vectors)}
        program_dict = {pid: vec for pid, vec in zip(program_ids, program_vectors)}
        
        differences = [
            np.linalg.norm(behavior_dict[pid] - program_dict[pid])
            for pid in common_ids
        ]
        
        return {
            'common_ids': list(common_ids),
            'behavior_only_ids': list(behavior_only_ids),
            'program_only_ids': list(program_only_ids),
            'differences': np.array(differences)
        }
    
    @staticmethod
    def visualize_results(behavior_vectors: np.ndarray, program_vectors: np.ndarray,
                         analysis_results: Dict[str, Any]) -> None:
        """Create visualizations of the analysis results.
        
        Args:
            behavior_vectors: Behavior encoding vectors
            program_vectors: Program encoding vectors
            analysis_results: Dictionary of analysis results
        """
        # Plot vector distance histogram
        plt.figure(figsize=(12, 6))
        plt.hist(analysis_results['differences'], bins=50, alpha=0.7, color='blue')
        plt.title('Distribution of Euclidean Distances Between Program and Behavior Vectors')
        plt.xlabel('Euclidean Distance')
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3)
        plt.savefig('vector_distance_histogram.png', dpi=300)
        plt.close()
        
        # Plot vector norms comparison
        behavior_norms = np.array([np.linalg.norm(vec) for vec in behavior_vectors])
        program_norms = np.array([np.linalg.norm(vec) for vec in program_vectors])
        
        plt.figure(figsize=(12, 6))
        plt.scatter(behavior_norms, program_norms, alpha=0.5)
        plt.plot([0, max(max(behavior_norms), max(program_norms))], 
                [0, max(max(behavior_norms), max(program_norms))], 
                'r--', label='y=x')
        plt.title('Comparison of Vector Norms: Program vs Behavior')
        plt.xlabel('Behavior Vector Norm')
        plt.ylabel('Program Vector Norm')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig('vector_norms_comparison.png', dpi=300)
        plt.close()
        
        # Perform PCA and plot results
        flat_behavior_vectors = behavior_vectors.reshape(behavior_vectors.shape[0], -1)
        flat_program_vectors = program_vectors.reshape(program_vectors.shape[0], -1)
        
        combined_data = np.vstack([flat_behavior_vectors, flat_program_vectors])
        pca = PCA(n_components=2)
        combined_pca = pca.fit_transform(combined_data)
        
        behavior_pca = combined_pca[:len(flat_behavior_vectors)]
        program_pca = combined_pca[len(flat_behavior_vectors):]
        
        plt.figure(figsize=(12, 10))
        plt.scatter(behavior_pca[:, 0], behavior_pca[:, 1], alpha=0.8, label='Behavior Encodings', color='blue')
        plt.scatter(program_pca[:, 0], program_pca[:, 1], alpha=0.8, label='Program Encodings', color='lightgreen')
        plt.title('Latent Space Encodings in 2D PCA Space')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig('pca_plot.png', dpi=300)
        plt.close()
        
        # Save results
        results = {
            'behavior_vectors': flat_behavior_vectors,
            'behavior_pca': behavior_pca,
            'program_vectors': flat_program_vectors,
            'program_pca': program_pca,
            'pca_explained_variance_ratio': pca.explained_variance_ratio_,
            **analysis_results,
            'behavior_norms': behavior_norms,
            'program_norms': program_norms
        }
        np.save('pca_plot.npy', results)


def main():
    """Main execution function."""
    try:
        # Initialize components
        dsl = get_DSL_option_v2(seed=42)
        encoder_manager = EncoderManager()
        data_processor = DataProcessor(dsl)
        
        # Load encoders
        behavior_encoder, program_encoder = encoder_manager.load_encoders()
        
        # Process HDF5 file
        hdf5_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/data.hdf5"
        programs = data_processor.process_hdf5_file(hdf5_file_path, behavior_encoder)
        
        # Encode behaviors
        behavior_vectors, behavior_ids = Encoder.encode_demos(programs, behavior_encoder)
        
        # Process text file
        txt_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/id.txt"
        program_data = data_processor.load_programs_from_txt(txt_file_path)
        
        # Encode programs
        program_vectors, program_ids = Encoder.encode_programs(program_data, program_encoder, dsl)
        
        # Analyze and visualize results
        if len(behavior_vectors) > 0 and len(program_vectors) > 0:
            analysis_results = Analyzer.analyze_encodings(
                behavior_vectors, program_vectors, behavior_ids, program_ids
            )
            Analyzer.visualize_results(behavior_vectors, program_vectors, analysis_results)
            
            # Print analysis summary
            print("\nAnalysis Summary:")
            print(f"Mean distance between vectors: {np.mean(analysis_results['differences']):.4f}")
            print(f"Standard deviation: {np.std(analysis_results['differences']):.4f}")
            
        else:
            print("No latent vectors were encoded. Check for errors above.")
            
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()