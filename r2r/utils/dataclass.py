import torch
from dataclasses import dataclass
from typing import List

@dataclass
class ModelOutputs:
    """
    Outputs from the model
    
    Args:
        logits: shape (batch_size, seq_len, vocab_size)
        hidden_states: shape (batch_size, seq_len, hidden_size), as a list of tensors, with the last item being the last layer
        token: shape (batch_size, seq_len), the token that was used to generate the output
    """
    logits: torch.Tensor
    hidden_states: List[torch.Tensor]
    token: torch.Tensor