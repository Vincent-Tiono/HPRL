import torch
from torch.autograd import Function

class VectorQuantization(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        Vector quantization for 2D inputs (batch of vectors)
        
        Args:
            inputs: shape [B, D] where B is batch size, D is embedding dimension
            codebook: shape [K, D] where K is codebook size, D is embedding dimension
            
        Returns:
            indices: shape [B] - index of the closest codebook vector for each input
        """
        with torch.no_grad():
            # Get dimensions
            embedding_size = codebook.size(1)  # D
            batch_size = inputs.size(0)  # B
            
            # Compute distances between each input vector and each codebook vector
            # L2 distances: ||x-c||² = ||x||² + ||c||² - 2x·c
            codebook_sqr = torch.sum(codebook ** 2, dim=1)  # [K]
            inputs_sqr = torch.sum(inputs ** 2, dim=1, keepdim=True)  # [B, 1]
            
            # Calculate pairwise distances: [B, K]
            distances = torch.addmm(
                codebook_sqr + inputs_sqr,  # [B, K]
                inputs,  # [B, D]
                codebook.t(),  # [D, K]
                alpha=-2.0, beta=1.0
            )
            
            # Find nearest embedding for each input vector
            _, indices = torch.min(distances, dim=1)  # [B]
            ctx.mark_non_differentiable(indices)
            
            return indices

    @staticmethod
    def backward(ctx, grad_output):
        raise RuntimeError('Trying to call `.grad()` on graph containing '
            '`VectorQuantization2D`. The function `VectorQuantization2D` '
            'is not differentiable. Use `VectorQuantizationStraightThrough2D` '
            'if you want a straight-through estimator of the gradient.')

class VectorQuantizationStraightThrough(Function):
    @staticmethod
    def forward(ctx, inputs, codebook):
        """
        Vector quantization with straight-through estimator for 2D inputs
        
        Args:
            inputs: shape [B, D] where B is batch size, D is embedding dimension
            codebook: shape [K, D] where K is codebook size, D is embedding dimension
            
        Returns:
            codes: Quantized vectors [B, D] (closest codebook entries)
            indices: Indices of closest codebook entries [B]
        """
        # Find nearest codebook entries
        indices = vq(inputs, codebook)  # [B]
        ctx.save_for_backward(indices, codebook)
        ctx.mark_non_differentiable(indices)
        
        # Get the actual codebook vectors for these indices
        codes = torch.index_select(codebook, dim=0, index=indices)  # [B, D]
        
        return (codes, indices)

    @staticmethod
    def backward(ctx, grad_output, grad_indices):
        """
        Straight-through estimator for vector quantization
        
        Args:
            grad_output: Gradient from downstream layers [B, D]
            grad_indices: Gradient for indices (always None) [B]
            
        Returns:
            grad_inputs: Gradient for encoder (straight-through) [B, D]
            grad_codebook: Gradient for codebook vectors [K, D]
        """
        grad_inputs, grad_codebook = None, None

        # Encoder learning: straight-through estimator
        # Pass gradients directly to encoder
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_output.clone()
            
        # Codebook learning
        if ctx.needs_input_grad[1]:
            indices, codebook = ctx.saved_tensors
            embedding_size = codebook.size(1)  # D
            
            # Initialize codebook gradients
            grad_codebook = torch.zeros_like(codebook)  # [K, D]
            
            # Accumulate gradients for each codebook vector
            # For each position where a codebook vector was used,
            # add the gradient at that position to the corresponding codebook entry
            grad_codebook.index_add_(0, indices, grad_output)

        return (grad_inputs, grad_codebook)

# Create callable functions
vq = VectorQuantization.apply
vq_st = VectorQuantizationStraightThrough.apply