import torch
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import traceback
import re
from pretrain.models_option_new_vae import ProgramEncoder
from karel_env.dsl import get_DSL_option_v2

def load_program_encoder():
    """
    Load the pre-trained program encoder from saved model weights.
    
    Returns:
        ProgramEncoder: The initialized and loaded program encoder model
    """
    # Load the trained weights
    params_list = torch.load('best_valid_params.ptp', map_location=torch.device('cpu'))
    param_dict = params_list[0]

    # Config for encoder initialization
    config = {
        'num_lstm_cell_units': 64,
        'net': {
            'rnn_type': 'GRU',
            'dropout': 0.0,
            'use_linear': True,
            'num_rnn_encoder_units': 256
        }
    }

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

    return program_encoder


def load_program_from_txt(id_file_path, dsl):
    """
    Load program IDs and their code from a text file.
    Ex. no_9269_prog_len_19_max_s_h_len_16 DEF run m( REPEAT...
    -> program_id = no_9269_prog_len_19_max_s_h_len_16
    -> program_code = DEF run m( REPEAT...
    -> program_data = [(program_id, program_code)]
    
    Args:
        id_file_path: Path to the text file containing program IDs and code
        dsl: The DSL object for parsing programs
        
    Returns:
        list: List of tuples (program_id, program_code)
    """
    program_data = []

    try:
        with open(id_file_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 1:
                    program_id = parts[0]
                    # If the line contains the program code
                    if len(parts) > 1:
                        program_code = ' '.join(parts[1:])
                    else:
                        program_code = None
                    program_data.append((program_id, program_code))
    except Exception as e:
        print(f"Error loading programs from {id_file_path}: {e}")
        traceback.print_exc()

    return program_data

def encode_programs(program_data, program_encoder, dsl):
    """
    Encode programs using the program encoder.
    
    Args:
        program_data: List of tuples (program_id, program_code)
        program_encoder: The program encoder model
        dsl: The DSL object for parsing programs
        
    Returns:
        tuple: (latent_vectors, program_ids) - Encoded vectors and corresponding program IDs
    """
    latent_vectors = []
    program_ids = []

    # Add tracking variables
    total_input_programs = len(program_data)
    programs_with_encoding = 0

    for program_id, program_code in program_data:
        if program_code is None:
            continue

        # Convert program to token
        program_tokens = dsl.str2intseq(program_code)

        # Convert to tensor, add batch dimension ex. [1, 2, 3] -> [[1, 2, 3]]
        # because program_encoder expects a batch dimension
        tokens_tensor = torch.LongTensor(program_tokens).unsqueeze(0)
        # Length of the program
        src_len = torch.LongTensor([len(program_tokens)])
        
        def _sample_latent(h_enc, hidden_size):
            """
            Return the latent normal sample z ~ N(mu, sigma^2)
            """
            _enc_mu = torch.nn.Linear(hidden_size, hidden_size)
            _enc_log_sigma = torch.nn.Linear(hidden_size, hidden_size)
            
            mu = _enc_mu(h_enc)
            log_sigma = _enc_log_sigma(h_enc)
            sigma = torch.exp(log_sigma)
            std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).to(torch.float).to(h_enc.device)
            
            return mu + sigma * torch.autograd.Variable(std_z, requires_grad=False) 

        # Encode program
        with torch.no_grad():
            # Get the output from the encoder
            _, encoder_output = program_encoder(tokens_tensor, src_len)
            
            z = _sample_latent(encoder_output.squeeze(), 64)
            
            latent_vectors.append(z.cpu().numpy())
            program_ids.append(program_id)
            programs_with_encoding += 1

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
    1. Loads the program encoder
    2. Processes TXT files to encode program source code
    3. Applies PCA for dimensionality reduction
    4. Visualizes the encodings in 2D space
    """
    try:
        # Initialize DSL for parsing programs
        seed = 42
        dsl = get_DSL_option_v2(seed=seed)

        # Load program encoder
        program_encoder = load_program_encoder()
        print("Successfully loaded program encoder")

        # Process txt files for program encoding
        all_program_vectors = []
        all_program_ids = []

        # Specify the TXT file path
        txt_file_path = "/tmp2/hubertchang/datasets_options_L30_1m_cover_branch/karel_dataset_option_L30_1m_cover_branch/id.txt"
        print(f"Processing {txt_file_path}")
        program_data = load_program_from_txt(txt_file_path, dsl)

        if program_data:
            print(f"Loaded {len(program_data)} programs from text file")
            # Process all programs
            latent_vectors, program_ids = encode_programs(program_data, program_encoder, dsl)

            if len(latent_vectors) > 0:
                all_program_vectors.extend(latent_vectors)
                all_program_ids.extend(program_ids)

        # Convert to numpy arrays
        if all_program_vectors:
            all_program_vectors = np.array(all_program_vectors)
            print(f"Encoded {len(all_program_vectors)} program samples")
            print(f"Program vectors shape: {all_program_vectors.shape}")

            # Flatten vectors for PCA if needed
            if len(all_program_vectors.shape) > 2:
                flat_program_vectors = all_program_vectors.reshape(all_program_vectors.shape[0], -1)
            else:
                flat_program_vectors = all_program_vectors

            # Fit PCA on program vectors to reduce to 2D
            pca = PCA(n_components=2)
            program_pca = pca.fit_transform(flat_program_vectors)

            # Generate 10,000 points from a normal distribution
            np.random.seed(42)
            normal_samples = np.random.normal(0, 1, size=(10000, 2))

            # Plot results
            plt.figure(figsize=(12, 10))
            
            # Plot program encodings
            plt.scatter(program_pca[:, 0], program_pca[:, 1], alpha=0.7, label='Program Encodings', color='orange')
            
            # Plot normal distribution points
            plt.scatter(normal_samples[:, 0], normal_samples[:, 1], alpha=0.3, label='Normal Distribution', color='blue', s=10)
            
            plt.title('Program Latent Space Encodings vs Normal Distribution')
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig('pca_plot.png', dpi=300)
            plt.close()

            # Save results to a file
            results = {
                'program_ids': all_program_ids,
                'program_vectors': flat_program_vectors,
                'program_pca': program_pca,
                'pca_explained_variance_ratio': pca.explained_variance_ratio_,
                'normal_samples': normal_samples
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