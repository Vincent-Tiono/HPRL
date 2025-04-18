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


def load_encoders():
    """
    Load the pre-trained behavior and program encoders from saved model weights.
    
    Returns:
        tuple: (behavior_encoder, program_encoder) - The initialized and loaded encoder models
    """
    # Load the trained weights
    params_list = torch.load('best_valid_params.ptp', map_location=torch.device('cpu'))
    param_dict = params_list[0]

    # Create config for behavior encoder
    config = {
        'recurrent_policy': True,
        'dsl': {'num_agent_actions': 6},
        'num_lstm_cell_units': 64,
        'net': {
            'rnn_type': 'GRU',
            'dropout': 0.0,
            'use_linear': True,
            'num_rnn_encoder_units': 256
        },
        'input_channel': 8,
        'input_height': 8,
        'input_width': 8
    }

    # Initialize behavior encoder
    behavior_encoder = BehaviorEncoder(
        recurrent=config['recurrent_policy'],
        num_actions=config['dsl']['num_agent_actions'],
        hidden_size=config['num_lstm_cell_units'],
        rnn_type=config['net']['rnn_type'],
        dropout=config['net']['dropout'],
        use_linear=config['net']['use_linear'],
        unit_size=config['net']['num_rnn_encoder_units'],
        input_channel=config['input_channel'],
        input_height=config['input_height'],
        input_width=config['input_width']
    )

    # Load behavior encoder weights
    behavior_encoder_state_dict = {}
    for key, value in param_dict.items():
        if key.startswith('vae.behavior_encoder'):
            new_key = key.replace('vae.behavior_encoder.', '')
            behavior_encoder_state_dict[new_key] = value

    behavior_encoder.load_state_dict(behavior_encoder_state_dict)
    behavior_encoder.eval()

    # Initialize program encoder
    program_encoder = ProgramEncoder(
        num_inputs=35,  # token space size
        num_outputs=64,  # latent space size
        recurrent=True,
        hidden_size=config['num_lstm_cell_units'],
        rnn_type=config['net']['rnn_type'],
        dropout=config['net']['dropout'],
        use_linear=config['net']['use_linear'],
        unit_size=config['net']['num_rnn_encoder_units']
    )

    # Load program encoder weights
    program_encoder_state_dict = {}
    for key, value in param_dict.items():
        if key.startswith('vae.program_encoder'):
            new_key = key.replace('vae.program_encoder.', '')
            program_encoder_state_dict[new_key] = value

    program_encoder.load_state_dict(program_encoder_state_dict)
    program_encoder.eval()

    return behavior_encoder, program_encoder

def get_exec_data(hdf5_file, program_id, num_agent_actions):
    """
    Extract execution data from the HDF5 file for a given program ID.
    
    Args:
        hdf5_file: Open HDF5 file containing program execution data
        program_id: ID of the program to extract data for
        num_agent_actions: Number of possible actions in the DSL
    
    Returns:
        tuple: (states, actions, action_lengths) - Numpy arrays of execution data
    """
    try:
        # First check if the program_id exists in the HDF5 file
        if program_id not in hdf5_file:
            print(f"Error: program_id '{program_id}' not found in the HDF5 file")
            return np.array([]), np.array([]), np.array([])

        # Check if s_h exists for this program
        if 's_h' not in hdf5_file[program_id]:
            print(f"Error: 's_h' field not found for program_id '{program_id}'")
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
                s_h[i][1, :, :, :] = s_h[i][0, :, :, :]
                a_h[i][0] = num_agent_actions - 1
                
        results = map(lambda x: np.expand_dims(x[0][0], 0), zip(s_h, s_h_len))
        s_h = np.stack(list(results))
        
        return s_h, a_h, a_h_len
    
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
        
        for program_id, s_h, a_h, a_h_len in programs:
            try:
                s_h_tensor = torch.FloatTensor(s_h)
                a_h_tensor = torch.LongTensor(a_h)
                
                if len(a_h_tensor.shape) == 1:
                    a_h_tensor = a_h_tensor.unsqueeze(0)
                    
                s_0_tensor = s_h_tensor.unsqueeze(0)
                a_h_tensor = a_h_tensor.unsqueeze(0)
                
                with torch.no_grad():
                    latent = behavior_encoder(s_0_tensor, a_h_tensor)
                    latent = latent.squeeze()
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

        # Encode program
        with torch.no_grad():
            # Get the output from the encoder
            _, encoder_output = program_encoder(tokens_tensor, src_len)
            
            z = _sample_latent(encoder_output.squeeze(), 64)
            
            latent_vectors.append(z.cpu().numpy())
            program_ids.append(program_id)


    # Print encoding statistics
    print(f"\nENCODING STATISTICS:")
    print(f"Input programs: {total_input_programs}")
    print(f"Programs successfully encoded: {programs_with_encoding}")
    print(f"Total encodings generated: {len(latent_vectors)}")


    # Convert list of arrays to single numpy array
    if latent_vectors:
        # Check if all vectors have the same shape
        shapes = [lv.shape for lv in latent_vectors]
        if len(set(shapes)) > 1:
            print(f"Warning: Different shapes in latent vectors: {set(shapes)}")
            # Find common dimension or pad/truncate as needed
            min_dim = min([lv.shape[1] for lv in latent_vectors])
            latent_vectors = [lv[:, :min_dim] for lv in latent_vectors]

        return np.vstack(latent_vectors), program_ids
    else:
        return np.array([]), []

def main():
    """
    Main function that orchestrates the encoding and visualization process.
    
    This function:
    1. Loads the encoders
    2. Processes HDF5 files to encode program behaviors
    3. Processes TXT files to encode program source code
    4. Combines the encodings with a normal distribution
    5. Applies PCA for dimensionality reduction
    6. Visualizes the encodings in 2D space
    7. Saves the results
    """
    try:
        # Initialize DSL for parsing programs
        seed = 42
        dsl = get_DSL_option_v2(seed=seed)

        # Load behavior and program encoders
        behavior_encoder, program_encoder = load_encoders()
        print("Successfully loaded encoders")

        # Process hdf5 files for behavior encoding
        all_behavior_vectors = []
        all_behavior_program_ids = []

        # Specify the HDF5 file path
        hdf5_file_path = "karel_dataset/data.hdf5"
        print(f"Processing {hdf5_file_path}")
        programs = process_hdf5_file(hdf5_file_path, behavior_encoder)

        if programs:
            latent_vectors, program_ids = encode_demos(programs, behavior_encoder)

            all_behavior_vectors.extend(latent_vectors)
            all_behavior_program_ids.extend(program_ids)

        # Process txt files for program encoding
        all_program_vectors = []
        all_program_ids = []

        # Specify the TXT file path
        txt_file_path = "karel_dataset/id.txt"
        print(f"Processing {txt_file_path}")
        program_data = load_program_from_txt(txt_file_path, dsl)

        if program_data:
            print(f"Loaded {len(program_data)} programs from text file")
            # Process all programs instead of just a subset
            latent_vectors, program_ids = encode_programs(program_data, program_encoder, dsl)

            if len(latent_vectors) > 0:
                all_program_vectors.extend(latent_vectors)
                all_program_ids.extend(program_ids)

        # Convert to numpy arrays
        if all_behavior_vectors and all_program_vectors:
            all_behavior_vectors = np.array(all_behavior_vectors)
            all_program_vectors = np.array(all_program_vectors)

            print(f"Encoded {len(all_behavior_vectors)} behavior samples")
            print(f"Behavior vectors shape: {all_behavior_vectors.shape}")

            print(f"Encoded {len(all_program_vectors)} program samples")
            print(f"Program vectors shape: {all_program_vectors.shape}")

            # Determine dimensions for normal distribution
            if len(all_behavior_vectors.shape) > 2:
                flat_behavior_shape = all_behavior_vectors.reshape(all_behavior_vectors.shape[0], -1).shape[1]
            else:
                flat_behavior_shape = all_behavior_vectors.shape[1]

            if len(all_program_vectors.shape) > 2:
                flat_program_shape = all_program_vectors.reshape(all_program_vectors.shape[0], -1).shape[1]
            else:
                flat_program_shape = all_program_vectors.shape[1]

            # Use larger dimension for normal distribution
            flat_shape = max(flat_behavior_shape, flat_program_shape)
            print(f"Using dimension {flat_shape} for normal distribution")

            # Generate normal distribution data for comparison
            num_samples = min(len(all_behavior_vectors) + len(all_program_vectors), 10000)  # Limit total samples
            normal_dist = np.random.normal(0, 1, (num_samples, flat_shape))
            print(f"Generated normal distribution with shape: {normal_dist.shape}")

            # Flatten vectors for PCA
            flat_behavior_vectors = all_behavior_vectors.reshape(all_behavior_vectors.shape[0], -1)
            flat_program_vectors = all_program_vectors.reshape(all_program_vectors.shape[0], -1)

            # Trim dimensions if needed to match
            min_dim = min(flat_behavior_shape, flat_program_shape, flat_shape)
            flat_behavior_vectors = flat_behavior_vectors[:, :min_dim]
            flat_program_vectors = flat_program_vectors[:, :min_dim]
            normal_dist = normal_dist[:, :min_dim]

            # Stack all data for PCA
            combined_data = np.vstack([flat_behavior_vectors, flat_program_vectors, normal_dist])

            # Fit PCA on combined data to reduce to 2D
            pca = PCA(n_components=2)
            combined_pca = pca.fit_transform(combined_data)

            # Split PCA results back into their original groups
            behavior_pca = combined_pca[:len(flat_behavior_vectors)]
            program_pca = combined_pca[len(flat_behavior_vectors):len(flat_behavior_vectors) + len(flat_program_vectors)]
            normal_pca = combined_pca[len(flat_behavior_vectors) + len(flat_program_vectors):]

            # Plot results - all three distributions
            plt.figure(figsize=(12, 10))

            # Plot behavior encodings
            plt.scatter(behavior_pca[:, 0], behavior_pca[:, 1], alpha=0.8, label='Behavior Encodings', color='lightblue')

            # Plot program encodings
            plt.scatter(program_pca[:, 0], program_pca[:, 1], alpha=0.8, label='Program Encodings', color='orange')

            # Plot normal distribution
            plt.scatter(normal_pca[:, 0], normal_pca[:, 1], alpha=0.05, label='Normal Distribution (0,1)', color='lightgreen')

            plt.title('Latent Space Encodings in 2D PCA Space')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('pca_plot.png', dpi=300)
            plt.close()

            # Save results to a file for further analysis
            results = {
                'behavior_program_ids': all_behavior_program_ids,
                'behavior_vectors': flat_behavior_vectors,
                'behavior_pca': behavior_pca,
                'program_ids': all_program_ids,
                'program_vectors': flat_program_vectors,
                'program_pca': program_pca,
                'normal_pca': normal_pca,
                'pca_explained_variance_ratio': pca.explained_variance_ratio_
            }
            np.save('pca_plot.npy', results)
            print("Results saved to pca_plot.npy")
        else:
            print("No latent vectors were encoded. Check for errors above.")
            
    except Exception as e:
        print(f"Error in main: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main() 