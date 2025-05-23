import torch
import torch.nn.functional as F
from typing import Tuple, Union

def compute_entropy(logits: torch.Tensor) -> Union[float, torch.Tensor]:
    """
    Calculate entropy of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        
    Returns:
        Entropy values as a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)  # [batch_size]
    
    return entropy.item() if is_single_input else entropy

def compute_logu(logits: torch.Tensor, topk: int = 10) -> Tuple[Union[float, torch.Tensor], Union[float, torch.Tensor]]:
    """
    Calculate log-u score of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider
        
    Returns:
        Tuple of (aleatoric_uncertainty, epistemic_uncertainty)
        Each is a scalar (if single input) or tensor of shape [batch_size]
    """
    # Handle single dimension input
    is_single_input = logits.dim() == 1
    if is_single_input:
        logits = logits.unsqueeze(0)
    
    # Get top-k logits and their indices
    topk_logits, topk_indices = torch.topk(logits, topk, dim=-1)  # [batch_size, topk]
    
    # Calculate sum of logits (S)
    alpha = torch.sum(topk_logits, dim=-1, keepdim=True)  # [batch_size, 1]
    
    # Calculate normalized probabilities (p_i = x_i/S)
    probs = topk_logits / alpha  # [batch_size, topk]
    
    # Calculate digamma terms
    digamma_xi = torch.digamma(topk_logits + 1)  # ψ(x_i + 1)
    digamma_sum = torch.digamma(alpha + 1)  # ψ(S + 1)
    
    # Calculate aleatoric uncertainty efficiently
    # AU = -∑(p_i * (ψ(x_i + 1) - ψ(S + 1)))
    aleatoric_uncertainty = -torch.sum(probs * (digamma_xi - digamma_sum), dim=-1)  # [batch_size]
    
    # Calculate epistemic uncertainty
    # EU = K / (S + K)
    epistemic_uncertainty = topk / (alpha.squeeze(-1) + topk)  # [batch_size]
    
    if is_single_input:
        return aleatoric_uncertainty.item(), epistemic_uncertainty.item()
    else:
        return aleatoric_uncertainty, epistemic_uncertainty

def compute_reliability(logits: torch.Tensor, topk: int = 10) -> Union[float, torch.Tensor]:
    """
    Calculate reliability of the prediction distribution.
    
    Args:
        logits: Unnormalized logits of shape [vocab_size] or [batch_size, vocab_size]
        topk: Number of top logits to consider
        
    Returns:
        Reliability values as a scalar (if single input) or tensor of shape [batch_size]
    """
    aleatoric_uncertainty, epistemic_uncertainty = compute_logu(logits, topk)
    
    # Handle both scalar and tensor inputs
    if isinstance(aleatoric_uncertainty, float):
        return 1 / (aleatoric_uncertainty * epistemic_uncertainty)
    else:
        return 1 / (aleatoric_uncertainty * epistemic_uncertainty)
