import torch
from typing import List, Dict, Any, Optional, Union
from transformers import PreTrainedTokenizer

class SGLangTokenManager:
    """
    Manages token sequences during text generation, including tracking active sequences,
    hidden states, and automatically handling sequence completion.
    """
    
    def __init__(
        self, 
        input_ids: List[List[int]], 
        tokenizer: PreTrainedTokenizer, 
        max_new_tokens: int = 128,
        record_hidden_states: bool = False,
        record_token_type: bool = False
    ) -> None:
        """
        Initialize the TokenManager with tokenized inputs.
        
        Args:
            input_ids (List[List[int]]): List of tokenized inputs
            tokenizer (PreTrainedTokenizer): Tokenizer for decoding
            max_new_tokens (int): Maximum number of tokens to generate
            record_hidden_states (bool): Whether to record hidden states during generation
            record_token_type (bool): Whether to record which model generated each token
        """
        self.input_ids = input_ids
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.max_new_tokens = max_new_tokens
        self.record_hidden_states = record_hidden_states
        self.record_token_type = record_token_type
            
        # Maintain separate lists for active and finished sequences
        self.active_sequences: List[Dict[str, Any]] = []
        self.finished_sequences: List[Dict[str, Any]] = []
        
        # Pre-allocate a list to directly reference current_ids from active sequences
        # This avoids creating a new list every time get_active_input_ids is called
        self.active_input_ids: List[List[int]] = []
        
        # Initialize sequences
        self.prepare_sequences()
        
    def prepare_sequences(self) -> None:
        """
        Process tokenized inputs into token sequences.
        """
        # Initialize sequences
        for idx, ids in enumerate(self.input_ids):
            # Create a mutable copy of input_ids that will be updated during generation
            current_ids = ids.copy()
            
            seq = {
                "original_index": idx,
                "input_ids": ids,
                "current_ids": current_ids,  # Will be updated during generation
                "output_ids": [],
                "is_finished": False
            }
            
            if self.record_hidden_states:
                seq["hidden_states"] = []
                
            if self.record_token_type:
                seq["token_types"] = []
                
            # All sequences start as active
            self.active_sequences.append(seq)
            
            # Store a direct reference to the current_ids list
            self.active_input_ids.append(current_ids)
    
    def get_active_input_ids(self) -> List[List[int]]:
        """
        Return token IDs for active sequences.
        
        Returns:
            List[List[int]]: List of token IDs for active sequences
        """
        # Return the pre-built list of references to current_ids
        # This avoids creating a new list with a list comprehension each time
        return self.active_input_ids
    
    def fetch_active_input_ids(self, index: List[int]) -> List[List[int]]:
        """
        Fetch active input ids for given indices.
        """
        return [self.active_input_ids[i] for i in index]
    
    def update_sequences_engine(self, outputs: List[Dict[str, Any]]) -> bool:
        """
        Process model outputs and update sequences.
        
        Args:
            outputs (List[Dict[str, Any]]): Output from llm.generate
            
        Returns:
            bool: True if any sequence finished in this update
        """
        generated_tokens = []
        hidden_states = []
        for output in outputs:
            generated_tokens.append(output["output_ids"][-1])
            if self.record_hidden_states and "meta_info" in output and "hidden_states" in output["meta_info"]:
                hidden_states.append(output["meta_info"]["hidden_states"][0][-1])
            else:
                hidden_states.append(None)

        return self.update_sequences_direct(generated_tokens, hidden_states)

    def update_sequences_direct(self, generated_tokens: List[int], hidden_states: List[torch.Tensor] | None = None, token_types: List[int] | None = None) -> bool:
        """
        Process model outputs and update sequences.
        
        Args:
            generated_tokens (List[int]): Generated tokens
            hidden_states (List[torch.Tensor]): Hidden states
            token_types (List[int]): Model type indicators (0: quick, 1: reference, etc.)
            
        Returns:
            bool: True if any sequence finished in this update
        """        
        any_finished = False
        indices_to_remove = []
        
        # Process all sequences and mark which ones to remove
        for i, seq in enumerate(self.active_sequences):
            # Get output for this sequence
            generated_token = generated_tokens[i]
            hidden_state = hidden_states[i] if hidden_states is not None else None
            token_type = token_types[i] if token_types is not None else None
            
            # Get hidden state (if available)
            if self.record_hidden_states and hidden_state is not None:
                seq["hidden_states"].append(hidden_state)
                
            # Record model type (if tracking is enabled)
            if self.record_token_type and token_type is not None:
                seq["token_types"].append(token_type)
            
            # Add the new token
            seq["output_ids"].append(generated_token)
            seq["current_ids"].append(generated_token)
            
            # Check if EOS token or reached max tokens
            if generated_token == self.eos_token_id or len(seq["output_ids"]) >= self.max_new_tokens:
                seq["is_finished"] = True
                self.finished_sequences.append(seq)
                indices_to_remove.append(i)
                any_finished = True
        
        # Truncate the active sequences to remove any finished ones
        if indices_to_remove:
            # Create new lists excluding the sequences at the marked indices
            self.active_sequences = [seq for i, seq in enumerate(self.active_sequences) if i not in indices_to_remove]
            self.active_input_ids = [ids for i, ids in enumerate(self.active_input_ids) if i not in indices_to_remove]
            
        return any_finished
    
    def is_generation_complete(self) -> bool:
        """
        Check if all sequences are finished.
        
        Returns:
            bool: True if all sequences are finished
        """
        return len(self.active_sequences) == 0
    
    def get_active_count(self) -> int:
        """
        Return the number of active sequences.
        
        Returns:
            int: Number of active sequences
        """
        return len(self.active_sequences)
    
    def get_active_index(self) -> List[int]:
        """
        Return the index of active sequences.
        """
        return [seq["original_index"] for seq in self.active_sequences]

    def get_final_outputs(self) -> List[Dict[str, Union[str, List[int], Optional[torch.Tensor]]]]:
        """
        Return completed sequences with all metadata.
        
        Returns:
            List[Dict[str, Union[str, List[int], Optional[torch.Tensor]]]]: List of dictionaries with sequence data
        """
        # Combine active and finished sequences
        all_sequences = self.finished_sequences + self.active_sequences
        
        # Sort by original index to maintain the same order as input
        sorted_sequences = sorted(all_sequences, key=lambda x: x["original_index"])
        
        final_outputs = []
        for seq in sorted_sequences:
            # Decode prompt from input_ids
            prompt = self.tokenizer.decode(seq["input_ids"])
                
            final_outputs.append({
                "prompt": prompt,
                "input_ids": seq["input_ids"],
                "output_ids": seq["output_ids"]
            })

            # Convert hidden states to tensor
            if self.record_hidden_states and "hidden_states" in seq and seq["hidden_states"]:
                hidden_states_tensor = torch.stack(seq["hidden_states"])
                final_outputs[-1]["hidden_states"] = hidden_states_tensor
                
            # Include model types if recorded
            if self.record_token_type and "token_types" in seq:
                final_outputs[-1]["token_types"] = seq["token_types"]
            
        return final_outputs
    
    def __len__(self) -> int:
        """
        Return number of sequences.
        
        Returns:
            int: Number of sequences
        """
        return len(self.active_sequences) + len(self.finished_sequences)
