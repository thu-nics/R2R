import torch
import torch.nn.functional as F
from typing import Union

def sample_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, top_k: int = -1, min_p: float = 0.0) -> Union[int, torch.Tensor]:
    """Sample a token from logits using temperature, top-p, top-k, and min-p sampling.
    Args:
        logits: Token logits of shape [vocab_size] or [batch_size, vocab_size]
        temperature: Temperature for sampling (>0). Higher values produce more random samples.
        top_p: Top-p probability threshold for nucleus sampling (0 < top_p ≤ 1)
        top_k: Top-k threshold for sampling (if -1, no top-k filtering is applied)
        min_p: Min-p threshold (0 ≤ min_p ≤ 1). Tokens with prob < min_p * max_prob are filtered out.
    Returns:
        Sampled token ID (int for single sample, tensor for batch)
    """
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    
    if logits.dim() not in [1, 2]:
        raise ValueError("logits must have shape [vocab_size] or [batch_size, vocab_size]")
        
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    batch_size = logits.shape[0]
    
    # For greedy sampling (temperature=0), just return argmax
    if temperature == 0 or temperature <= 1e-5:
        tokens = torch.argmax(logits, dim=-1)
        return tokens.item() if is_single_input else tokens
    
    # Convert to probabilities
    probs = torch.nn.functional.softmax(logits / temperature, dim=-1)
    
    # Apply top-k filtering first (if specified)
    if top_k != -1:
        # Get top-k values and indices
        top_k_values, top_k_indices = torch.topk(probs, k=min(top_k, probs.shape[-1]), dim=-1)
        
        # Create a mask to zero out non-top-k probabilities
        mask = torch.zeros_like(probs, dtype=torch.bool)
        mask.scatter_(-1, top_k_indices, True)
        
        # Zero out non-top-k probabilities
        probs = probs * mask.float()
        
        # Renormalize probabilities
        probs = probs / probs.sum(dim=-1, keepdim=True)
    
    # Apply min-p filtering: remove tokens with prob < min_p * max_prob
    if min_p > 0.0:
        max_probs = probs.max(dim=-1, keepdim=True).values
        min_p_mask = probs >= (min_p * max_probs)
        min_p_mask[:, probs.argmax(dim=-1)] = True  # always keep the top token
        probs = probs * min_p_mask.float()
        probs = probs / probs.sum(dim=-1, keepdim=True)

    # Apply top-p (nucleus) sampling
    if top_p < 1.0:
        # Sort probabilities in descending order
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        
        # Calculate cumulative probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Create a mask for probabilities to keep
        # Values above top_p threshold are masked out
        mask = cumulative_probs <= top_p
        
        # Always keep at least one token
        mask[:, 0] = True
        
        # Zero out masked positions to exclude them from sampling
        sorted_probs = sorted_probs * mask.float()
        
        # Renormalize probabilities
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        
        # Sample from the filtered distribution
        sampled_indices = torch.multinomial(sorted_probs, num_samples=1)
        
        # Map back to original vocabulary indices
        tokens = torch.gather(sorted_indices, dim=-1, index=sampled_indices)
        tokens = tokens.squeeze(-1)  # Remove sample dimension
    else:
        # Direct sampling if no top-p filtering
        tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return tokens.item() if is_single_input else tokens
