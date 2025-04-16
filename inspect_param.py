import os
import sys
import torch
import numpy as np
import argparse
from collections import defaultdict

def inspect_params(model_path, summary=True, detailed=False, histograms=False, latent_vectors=False):
    """
    Inspect the parameters of a trained model saved in .ptp format
    
    Args:
        model_path: Path to the saved model parameters
        summary: Print summary information
        detailed: Print detailed parameter information
        histograms: Generate histograms of parameter distributions
        latent_vectors: If available, display information about latent vectors
    """
    print(f"Loading model parameters from: {model_path}")
    
    # Check if file exists
    if not os.path.exists(model_path):
        print(f"Error: File {model_path} does not exist")
        return
    
    try:
        # Load the model parameters
        params = torch.load(model_path, map_location=torch.device('cpu'))
        
        if isinstance(params, list):
            print(f"Model params are stored as a list of length {len(params)}")
            state_dict = params[0]  # Usually the first element is the model state dict
            
            # Check if there's an ob_rms (observation normalization stats)
            if len(params) > 1 and params[1] is not None:
                print("\nNormalization statistics (ob_rms):")
                print(f"  Mean: {params[1].mean}")
                print(f"  Variance: {params[1].var}")
        else:
            state_dict = params
        
        # Print model summary information
        if summary:
            total_params = 0
            total_size_bytes = 0
            
            # Group parameters by module
            module_params = defaultdict(int)
            
            for name, param in state_dict.items():
                num_params = np.prod(param.shape)
                total_params += num_params
                
                # Group by top-level module
                module_name = name.split('.')[0]
                module_params[module_name] += num_params
                
                param_size_bytes = param.element_size() * num_params
                total_size_bytes += param_size_bytes
            
            print("\n=== Model Summary ===")
            print(f"Total number of parameters: {total_params:,}")
            print(f"Total size: {total_size_bytes / (1024 * 1024):.2f} MB")
            
            print("\n=== Parameters by Module ===")
            for module, count in sorted(module_params.items(), key=lambda x: x[1], reverse=True):
                print(f"{module}: {count:,} parameters ({count/total_params*100:.2f}%)")
        
        # Print all parameter keys
        print("\n=== Parameter Keys ===")
        for i, key in enumerate(sorted(state_dict.keys())):
            print(f"{i+1}. {key} (shape: {state_dict[key].shape})")
        
        # Process and print details for each parameter
        if detailed:
            print("\n=== Detailed Parameter Information ===")
            param_info = []
            
            for name, param in state_dict.items():
                # Calculate stats
                num_params = np.prod(param.shape)
                
                param_size_bytes = param.element_size() * num_params
                
                param_info.append({
                    'name': name,
                    'shape': param.shape,
                    'dtype': param.dtype,
                    'num_params': num_params,
                    'size_kb': param_size_bytes / 1024,
                    'min': param.min().item() if param.numel() > 0 else None,
                    'max': param.max().item() if param.numel() > 0 else None,
                    'mean': param.mean().item() if param.numel() > 0 else None,
                    'std': param.std().item() if param.numel() > 0 else None
                })
            
            # Print detailed information
            for info in param_info:
                print(f"\nParameter: {info['name']}")
                print(f"  Shape: {info['shape']}")
                print(f"  Data type: {info['dtype']}")
                print(f"  Number of elements: {info['num_params']:,}")
                print(f"  Size: {info['size_kb']:.2f} KB")
                if info['min'] is not None:
                    print(f"  Min value: {info['min']:.6f}")
                    print(f"  Max value: {info['max']:.6f}")
                    print(f"  Mean value: {info['mean']:.6f}")
                    print(f"  Std. deviation: {info['std']:.6f}")
                    
                # Calculate and print histograms if requested
                if histograms and info['num_params'] > 0:
                    try:
                        import matplotlib.pyplot as plt
                        param_data = state_dict[info['name']].flatten().cpu().numpy()
                        plt.figure(figsize=(10, 4))
                        plt.hist(param_data, bins=50)
                        plt.title(f"Histogram for {info['name']}")
                        plt.savefig(f"{info['name'].replace('.', '_')}_histogram.png")
                        plt.close()
                        print(f"  Histogram saved as {info['name'].replace('.', '_')}_histogram.png")
                    except ImportError:
                        print("  Matplotlib not available for histogram generation")
        
        # Analyze the model architecture
        print("\n=== Model Architecture ===")
        model_components = set()
        for name in state_dict.keys():
            parts = name.split('.')
            for i in range(1, len(parts)):
                model_components.add('.'.join(parts[:i]))
        
        # Print model architecture hierarchy
        last_level = 0
        for component in sorted(model_components):
            level = component.count('.')
            indent = "  " * level
            print(f"{indent}├─ {component.split('.')[-1]}")
        
    except Exception as e:
        print(f"Error inspecting model parameters: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect trained model parameters")
    parser.add_argument("model_path", help="Path to the model parameters file (.ptp)")
    parser.add_argument("--summary", action="store_true", default=True, 
                        help="Print summary information (default: True)")
    parser.add_argument("--detailed", action="store_true", default=False, 
                        help="Print detailed parameter information")
    parser.add_argument("--histograms", action="store_true", default=False, 
                        help="Generate histograms of parameter distributions (requires matplotlib)")
    parser.add_argument("--latent-vectors", action="store_true", default=False, 
                        help="Display information about latent vectors if available")
    
    args = parser.parse_args()
    
    inspect_params(
        args.model_path, 
        summary=args.summary,
        detailed=args.detailed,
        histograms=args.histograms,
        latent_vectors=args.latent_vectors
    ) 