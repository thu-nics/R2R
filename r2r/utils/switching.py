import torch
from typing import List, Optional
import numpy as np
from collections import deque
from dataclasses import dataclass
import os
import sys
import random

from r2r.utils.metrics import compute_logu, compute_entropy
from r2r.models.router import load_model
from r2r.utils.dataclass import ModelOutputs

@dataclass
class SwitchingState:
    """State information for model switching decisions"""
    last_model: str  # 'quick' or 'reference'
    consecutive_simple_tokens: int = 0
    aleatoric_history: List[float] = None
    critical_history: List[float] = None
    momentum: float = 0.0

class ModelSwitchingStrategy:
    """Base class for model switching strategies"""
    def __init__(self, aleatoric_threshold: float = 2.275):
        self.aleatoric_threshold = aleatoric_threshold

        # self.entropy_threshold = 0.35
        # self.aleatoric_threshold = 2.250
        # self.epistemic_threshold = 0.0656562983380584
        self.state = SwitchingState(last_model='reference')
    
    def route(self, outputs: ModelOutputs) -> str:
        """Route to appropriate model and update state
        
        Args:
            outputs: Model outputs containing logits for uncertainty computation
            
        Returns:
            str: 'quick' or 'reference' indicating which model to use
        """
        raise NotImplementedError

class ImmediateSwitching(ModelSwitchingStrategy):
    """Simple immediate switching based on threshold"""
    def route(self, outputs) -> str:
        next_token_logits = outputs.logits[:, -1, :]
        aleatoric_uncertainty, _ = compute_logu(next_token_logits)
        model_choice = 'quick' if aleatoric_uncertainty < self.aleatoric_threshold else 'reference'
        self.state.last_model = model_choice
        return model_choice

class MomentumSwitching(ModelSwitchingStrategy):
    """Momentum-based switching with asymmetric behavior"""
    def __init__(self, aleatoric_threshold: float = 2.275, 
                 momentum_factor: float = 0.7,
                 quick_to_ref_threshold: float = 0.3,
                 ref_to_quick_threshold: float = 0.7):
        super().__init__(aleatoric_threshold)
        self.momentum_factor = momentum_factor
        self.quick_to_ref_threshold = quick_to_ref_threshold
        self.ref_to_quick_threshold = ref_to_quick_threshold
        self.state.momentum = 0.0
    
    def route(self, outputs) -> str:
        next_token_logits = outputs.logits[:, -1, :]
        aleatoric_uncertainty, _ = compute_logu(next_token_logits)
        is_simple = aleatoric_uncertainty < self.aleatoric_threshold
        
        # Update momentum based on current token
        self.state.momentum = (self.momentum_factor * self.state.momentum + 
                             (1 - self.momentum_factor) * (1.0 if is_simple else 0.0))
        
        if self.state.last_model == 'quick':
            model_choice = 'quick' if self.state.momentum > self.quick_to_ref_threshold else 'reference'
        else:
            model_choice = 'quick' if self.state.momentum > self.ref_to_quick_threshold else 'reference'
        
        self.state.last_model = model_choice
        return model_choice

class SingleRollingWindowSwitching(ModelSwitchingStrategy):
    """Rolling window-based switching with asymmetric behavior"""
    def __init__(self, aleatoric_threshold: float = 2.275,
                 epistemic_threshold: float = 0.055,
                 entropy_threshold: float = 0.02,
                 window_size: int = 3,
                 required_simple_ratio: float = 1.0):
        super().__init__(aleatoric_threshold)
        self.epistemic_threshold = epistemic_threshold
        self.entropy_threshold = entropy_threshold
        self.window_size = window_size
        self.required_simple_ratio = required_simple_ratio
        self.state.aleatoric_history = deque(maxlen=window_size)
    
    def route(self, outputs) -> str:
        next_token_logits = outputs.logits[:, -1, :]
        aleatoric_uncertainty, epistemic_uncertainty = compute_logu(next_token_logits)
        entropy = compute_entropy(next_token_logits)
        
        is_simple = aleatoric_uncertainty < self.aleatoric_threshold or entropy < self.entropy_threshold
        if self.state.last_model == 'quick':
            model_choice = 'quick' if is_simple else 'reference'
            if model_choice == 'reference':
                self.state.aleatoric_history.clear()
                self.state.aleatoric_history.append(aleatoric_uncertainty)
        else:
            # Reference model: record history and check average uncertainty
            self.state.aleatoric_history.append(aleatoric_uncertainty)
            if len(self.state.aleatoric_history) == 0:
                model_choice = 'reference'
            else:
                avg_uncertainty = sum(self.state.aleatoric_history) / len(self.state.aleatoric_history)
                model_choice = 'quick' if avg_uncertainty < self.aleatoric_threshold else 'reference'
            
            # Clear history when switching back to quick
            if model_choice == 'quick':
                self.state.aleatoric_history.clear()
        
        self.state.last_model = model_choice
        return model_choice

class DuoRollingWindowSwitching(ModelSwitchingStrategy):
    """Rolling window-based switching with separate windows for quick and reference models"""
    def __init__(self, aleatoric_threshold: float = 2.275,
                 window_size: int = 3,
                 required_simple_ratio: float = 1.0):
        super().__init__(aleatoric_threshold)
        self.window_size = window_size
        self.required_simple_ratio = required_simple_ratio
        # Initialize separate windows for each model
        self.quick_history = deque(maxlen=window_size)
        self.reference_history = deque(maxlen=window_size)

    def route(self, outputs) -> str:
        next_token_logits = outputs.logits[:, -1, :]
        aleatoric_uncertainty, _ = compute_logu(next_token_logits)
        is_simple = aleatoric_uncertainty < self.aleatoric_threshold

        # Always record uncertainty in the current model's window
        if self.state.last_model == 'quick':
            self.quick_history.append(aleatoric_uncertainty)

            # If current token is complex, switch to reference immediately
            if not is_simple:
                self.reference_history.clear()  # Clear reference history when switching
                model_choice = 'reference'
            else:
                # Stay with quick model
                model_choice = 'quick'
        else:  # In reference model
            self.reference_history.append(aleatoric_uncertainty)

            # Only consider switching to quick if we have enough history
            if len(self.reference_history) > 0:
                # Check average uncertainty in reference window
                ref_avg = sum(self.reference_history) / len(self.reference_history)
                if ref_avg < self.aleatoric_threshold:
                    self.quick_history.clear()  # Clear quick history when switching
                    model_choice = 'quick'
                else:
                    model_choice = 'reference'
            else:
                model_choice = 'reference'

        self.state.last_model = model_choice
        return model_choice


class NeuralSwitching(ModelSwitchingStrategy):
    """Neural network-based switching using a trained critical case classifier"""

    def __init__(
        self, model_path, threshold: Optional[float] = None, device: str = "cuda", dtype=torch.float32
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # Load model using the load_model function from classifier.py
        self.model, model_config = load_model(model_path, device=self.device)

        # Use saved optimal threshold if available in common_args
        if threshold is None:
            self.threshold = float(model_config["common_args"]["threshold"])
            print(f"Using saved optimal threshold: {self.threshold}")
        else:
            self.threshold = float(threshold)
            print(f"Using provided threshold: {self.threshold}")

        # Extract model parameters
        self.init_args = model_config["init_args"]
        self.common_args = model_config["common_args"]
        self.logits_size = self.init_args.get("logits_size", 0)

        # Determine input type from common_args
        self.input_type = self.common_args["input_type"]
        if not isinstance(self.input_type, list):
            self.input_type = [self.input_type]
        self.model_type = model_config["model_type"]

        print(f"Using input types: {self.input_type}")

        # Set model to evaluation mode
        self.model.eval()

    def route(self, outputs: ModelOutputs) -> torch.Tensor:
        """
        Determine which model to use for each input in the batch.
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        with torch.no_grad():
            # Get batch size from outputs
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            batch_size = next_token_logits.shape[0]
            
            # Prepare inputs based on input_type
            inputs = {}
            
            # Process logits if needed
            if "logits" in self.input_type:
                # If the model has a logits_size parameter, use it to get top-k logits
                if self.logits_size > 0:
                    top_logits, _ = torch.topk(
                        next_token_logits, k=self.logits_size, dim=-1
                    )
                    inputs["logits"] = top_logits.to(
                        device=self.device, dtype=torch.float32
                    )  # [batch_size, topk]
                else:
                    # If no logits_size, use all logits
                    inputs["logits"] = next_token_logits.to(
                        device=self.device, dtype=torch.float32
                    )
            
            # Process hidden states if needed
            if "hidden_states" in self.input_type:
                inputs["hidden_states"] = outputs.hidden_states[-1][:, -1, :].to(
                    device=self.device, dtype=torch.float32
                )
            
            # Process token IDs if needed
            if "token" in self.input_type:
                inputs["token"] = outputs.token[:, -1].to(
                    device=self.device, dtype=torch.long
                )

            # Forward pass through the model with appropriate inputs
            model_output = self.model(**inputs)
            
            # Handle different output formats (single output or multi-class)
            if model_output.shape[1] == 1:
                critical_prob = torch.sigmoid(model_output).squeeze(-1)  # [batch_size]
                # Convert probabilities to binary decisions (0 = quick, 1 = reference)
                model_choices = (critical_prob >= self.threshold).to(torch.int)
            else:
                # For multi-class output, consider class 2 as critical (divergent) cases
                # Classes: 0=match, 1=mismatch, 2=divergent
                probabilities = torch.softmax(model_output, dim=1)  # [batch_size, num_classes]
                critical_prob = probabilities[:, 2]  # Get probability of class 2 (divergent)
                model_choices = (critical_prob >= self.threshold).to(torch.int)
            
            # For tracking state, we'll keep the most recent decision for each input
            self.state.last_model = "reference" if model_choices.any().item() else "quick"
            
            return model_choices


class NeuralRollingWindowSwitching(ModelSwitchingStrategy):
    """Neural network-based switching using a trained critical case classifier with rolling window"""

    def __init__(
        self,
        model_path: str = "critical_classifier_0227.pt",
        window_size: int = 3,
        required_simple_ratio: float = 1.0,
        threshold: Optional[float] = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype

        # Load model using the load_model function from classifier.py
        self.model, model_config = load_model(model_path, device=self.device)

        # Use saved optimal threshold if available in common_args
        if threshold is None:
            self.threshold = float(model_config["common_args"]["threshold"])
            print(f"Using saved optimal threshold: {self.threshold}")
        else:
            self.threshold = float(threshold)
            print(f"Using provided threshold: {self.threshold}")

        # Extract model parameters
        self.init_args = model_config["init_args"]
        self.common_args = model_config["common_args"]
        self.logits_size = self.init_args.get("logits_size", 0)

        # Determine input type from common_args
        self.input_type = self.common_args["input_type"]
        self.model_type = model_config["model_type"]

        print(f"Using input type: {self.input_type}")

        # Set window parameters
        self.window_size = window_size
        self.required_simple_ratio = required_simple_ratio
        self.critical_history = deque(maxlen=self.window_size)

        # Set model to evaluation mode
        self.model.eval()

    def route(self, outputs) -> str:
        with torch.no_grad():
            # Get top k logits
            next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]

            # Process logits if needed
            if self.input_type in ["logits", "both"]:
                # If the model has a logits_size parameter, use it to get top-k logits
                if self.logits_size > 0:
                    top_logits, _ = torch.topk(
                        next_token_logits, k=self.logits_size, dim=-1
                    )
                    top_logits = top_logits.to(
                        device=self.device, dtype=torch.float32
                    )  # [batch_size, topk]
                else:
                    # If no logits_size, use all logits
                    top_logits = next_token_logits.to(
                        device=self.device, dtype=torch.float32
                    )
            else:
                top_logits = None

            # Process hidden states if needed
            if self.input_type in ["hidden_states", "both"]:
                last_hidden_state = outputs.hidden_states[-1][:, -1, :].to(
                    device=self.device, dtype=torch.float32
                )
            else:
                last_hidden_state = None

            # Forward pass through the model based on input_type
            if self.input_type == "logits":
                critical_prob = torch.nn.functional.sigmoid(
                    self.model(top_logits)
                ).squeeze()
            elif self.input_type == "hidden_states":
                critical_prob = torch.nn.functional.sigmoid(
                    self.model(last_hidden_state)
                ).squeeze()
            elif self.input_type == "both":
                critical_prob = torch.nn.functional.sigmoid(
                    self.model(top_logits, last_hidden_state)
                ).squeeze()
            else:
                raise ValueError(f"Unsupported input_type: {self.input_type}")

            # Determine if the token is divergent/critical
            is_divergent = critical_prob >= self.threshold

            # Apply rolling window logic
            if self.state.last_model == "quick":
                model_choice = "quick" if not is_divergent else "reference"
                if model_choice == "reference":
                    self.critical_history.clear()
                    self.critical_history.append(critical_prob)
            else:
                # Reference model: record history and check average uncertainty
                self.critical_history.append(critical_prob)
                if len(self.critical_history) == 0:
                    model_choice = "reference"
                else:
                    avg_critical_prob = sum(self.critical_history) / len(self.critical_history)
                    model_choice = "quick" if avg_critical_prob < self.threshold else "reference"

                # Clear history when switching back to quick
                if model_choice == "quick":
                    self.critical_history.clear()

            self.state.last_model = model_choice
            return model_choice

class NeuralMultiInputSwitching(ModelSwitchingStrategy):
    """Neural network-based switching using a trained critical case classifier"""
    def __init__(self, model_path: str = 'critical_classifier_multi_input_0304.pt',
                 neural_window_size: int = 3,
                 threshold: Optional[float] = None,
                 device: str = 'cuda',
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        
        raise NotImplementedError

        self.device = device
        self.dtype = dtype
        # Load model using the load_model function from classifier.py
        self.model, model_config = load_model(model_path, device=self.device)
        
        # Use saved optimal threshold if available in common_args
        if 'threshold' in model_config['common_args'] and model_config['common_args']['threshold'] is not None and threshold is None:
            self.threshold = float(model_config['common_args']['threshold'])
            print(f"Using saved optimal threshold: {self.threshold}")
        else:
            self.threshold = float(threshold) if threshold is not None else 0.5
            print(f"Using provided threshold: {self.threshold}")
        
        # Extract model parameters
        self.init_args = model_config['init_args']
        self.common_args = model_config['common_args']
        self.logits_size = self.init_args.get('logits_size', 0)
        self.hidden_states_size = self.init_args.get('hidden_states_size', 0)
        
        # Determine input type from common_args or model type
        self.input_type = self.common_args.get('input_type', 'logits')
        self.model_type = model_config['model_type']
        
        # For backward compatibility, also check model type
        if self.input_type == 'logits' and 'HiddenStates' in self.model_type:
            self.input_type = 'hidden_states'
        elif self.input_type == 'logits' and 'LogitsHiddenStates' in self.model_type:
            self.input_type = 'both'
        
        print(f"Using input type: {self.input_type}")
        
        # Get neural window size from model config or use provided value
        self.neural_window_size = self.init_args.get('neural_window_size', neural_window_size)
        
        # Initialize queues for storing token information
        self.output_logits_queue = deque(maxlen=self.neural_window_size)
        self.output_hidden_states_queue = deque(maxlen=self.neural_window_size)
        
        # Set model to evaluation mode
        self.model.eval()

    def route(self, outputs) -> str:
        with torch.no_grad():
            # Get the last token's logits and hidden states
            batch_size, seq_len, vocab_size = outputs.logits.size()
            
            if seq_len != 1:  # If the current output length is not 1, reset queues (prefill stage)
                self.output_logits_queue.clear()
                self.output_hidden_states_queue.clear()
            
            # Process logits if needed
            if self.input_type in ['logits', 'both']:
                # If logits_size is specified, get top-k logits
                if self.logits_size > 0:
                    top_logits, _ = torch.topk(outputs.logits[:, -1:, :], 
                                              k=self.logits_size // self.neural_window_size, 
                                              dim=-1)  # [batch_size, 1, topk]
                else:
                    # Otherwise use all logits
                    top_logits = outputs.logits[:, -1:, :]
                
                # Add to queue
                self.output_logits_queue.append(top_logits)
            
            # Process hidden states if needed
            if self.input_type in ['hidden_states', 'both']:
                last_hidden_states = outputs.hidden_states[-1][:, -1:, :].to(device=self.device, dtype=torch.float32)
                # Add to queue
                self.output_hidden_states_queue.append(last_hidden_states)
            
            # If we don't have enough tokens yet, default to reference model
            if (self.input_type in ['logits', 'both'] and len(self.output_logits_queue) < self.neural_window_size) or \
               (self.input_type in ['hidden_states', 'both'] and len(self.output_hidden_states_queue) < self.neural_window_size):
                self.state.last_model = 'reference'
                return 'reference'
            
            # Prepare inputs based on model type and input_type
            if 'Multi' in self.model_type:
                # For multi-input models, concatenate the window of tokens
                if self.input_type in ['logits', 'both']:
                    logits_tensor = torch.cat(list(self.output_logits_queue), dim=1)
                    # Reshape for multi-logits models
                    logits_tensor = logits_tensor.view(batch_size, -1)  # Flatten to [batch_size, neural_window_size * topk]
                else:
                    logits_tensor = None
                
                if self.input_type in ['hidden_states', 'both']:
                    hidden_states_tensor = torch.cat(list(self.output_hidden_states_queue), dim=1)
                    # Reshape for multi-hidden-states models
                    hidden_states_tensor = hidden_states_tensor.view(batch_size, -1)  # Flatten to [batch_size, neural_window_size * hidden_size]
                else:
                    hidden_states_tensor = None
            else:
                # For single-token models, just use the latest token
                if self.input_type in ['logits', 'both']:
                    logits_tensor = self.output_logits_queue[-1].squeeze(1)  # [batch_size, topk]
                else:
                    logits_tensor = None
                
                if self.input_type in ['hidden_states', 'both']:
                    hidden_states_tensor = self.output_hidden_states_queue[-1].squeeze(1)  # [batch_size, hidden_size]
                else:
                    hidden_states_tensor = None
            
            # Apply softmax normalization to logits if needed
            if logits_tensor is not None and hasattr(self.model, 'normalize_input') and getattr(self.model, 'normalize_input', False):
                if 'Multi' in self.model_type and 'Logits' in self.model_type:
                    # For MultiLogitsClassifier, reshape to apply softmax correctly
                    batch_size = logits_tensor.shape[0]
                    single_logit_size = logits_tensor.shape[1] // self.neural_window_size
                    reshaped_logits = logits_tensor.view(batch_size, self.neural_window_size, single_logit_size)
                    normalized_logits = torch.nn.functional.softmax(reshaped_logits, dim=-1)
                    logits_tensor = normalized_logits.reshape(batch_size, -1)
                else:
                    logits_tensor = torch.nn.functional.softmax(logits_tensor, dim=-1)
            
            # Forward pass through the model based on input_type
            if self.input_type == 'logits':
                critical_prob = torch.nn.functional.sigmoid(self.model(logits_tensor)).squeeze()
            elif self.input_type == 'hidden_states':
                critical_prob = torch.nn.functional.sigmoid(self.model(hidden_states_tensor)).squeeze()
            elif self.input_type == 'both':
                critical_prob = torch.nn.functional.sigmoid(self.model(logits_tensor, hidden_states_tensor)).squeeze()
            else:
                raise ValueError(f"Unsupported input_type: {self.input_type}")
            
            # Determine if the token is simple or complex
            is_simple = (critical_prob < self.threshold).item()
            
            model_choice = 'quick' if is_simple else 'reference'
            self.state.last_model = model_choice
            return model_choice

class RandomSwitching(ModelSwitchingStrategy):
    """Random switching strategy that selects reference model with a given probability"""
    
    def __init__(self, reference_prob: float = 0.5, random_seed: Optional[int] = 42):
        """Initialize random switching strategy
        
        Args:
            reference_prob: Probability of selecting the reference model (0.0 to 1.0)
            random_seed: Optional random seed for reproducibility
        """
        super().__init__()
        self.reference_prob = reference_prob
        
        # Set random seed
        random.seed(random_seed)
            
        print(f"Initialized RandomSwitching with reference_prob={reference_prob}, random_seed={random_seed}")
    
    def route(self, outputs) -> torch.Tensor:
        """Randomly select between quick and reference models
        
        Args:
            outputs: Model outputs from the quick model
        Returns:
            torch.Tensor: Binary tensor of shape [batch_size] where:
                0 = use quick model
                1 = use reference model
        """
        # Get batch size from outputs
        batch_size = outputs.logits.size(0)
        
        # Generate random values for each item in batch
        rand_vals = torch.rand(batch_size, device=outputs.logits.device)
        
        # Convert to binary decisions (0 = quick, 1 = reference)
        model_choices = (rand_vals < self.reference_prob).to(torch.int)
        
        # Update state with most recent decision
        self.state.last_model = "reference" if model_choices.any().item() else "quick"
        
        return model_choices

def create_switching_strategy(strategy_name: str, **kwargs) -> ModelSwitchingStrategy:
    """Factory function to create switching strategy instances"""
    strategies = {
        'immediate': ImmediateSwitching,
        'momentum': MomentumSwitching,
        'rolling': SingleRollingWindowSwitching,
        'duo_rolling': DuoRollingWindowSwitching,
        'neural': NeuralSwitching,
        'neural_rolling': NeuralRollingWindowSwitching,
        'neural_multi_input': NeuralMultiInputSwitching,
        'random': RandomSwitching
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. "
                        f"Available strategies: {list(strategies.keys())}")
    
    return strategies[strategy_name](**kwargs) 
