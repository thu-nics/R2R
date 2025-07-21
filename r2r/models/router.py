import numpy as np
import torch
import torch.nn as nn
from collections import deque
from sklearn.metrics import confusion_matrix
import json
import pandas as pd
import math
import inspect
from transformers import AutoModelForCausalLM
from typing import List

################ Model Registry, Saving and Loading #################
MODEL_REGISTRY = {}


def register_model(cls=None, name=None):
    """
    Register a model class in the global registry.
    Can be used as a decorator with or without arguments.
    
    Args:
        cls: The class to register
        name: Optional name to register the class under. If None, uses the class name.
        
    Returns:
        The registered class
    """

    def _register(cls):
        model_name = name if name is not None else cls.__name__
        MODEL_REGISTRY[model_name] = cls
        # Also register with lowercase name for case-insensitive lookup
        MODEL_REGISTRY[model_name.lower()] = cls
        return cls
    
    # Called as @register_model
    if cls is not None:
        return _register(cls)
    
    # Called as @register_model() or @register_model(name="ModelName")
    return _register


def capture_init_args(cls):
    """
    Decorator to capture initialization arguments of a model class.
    
    Args:
        cls: The class to decorate
        
    Returns:
        The decorated class with automatic init args capture
    """
    original_init = cls.__init__
    
    def new_init(self, *args, **kwargs):
        # Store all initialization arguments
        self._init_args = {}
        
        # Get parameter names from the original __init__ method
        sig = inspect.signature(original_init)
        param_names = list(sig.parameters.keys())[1:]  # Skip 'self'
        
        # Map positional args to parameter names
        for i, arg in enumerate(args):
            if i < len(param_names):
                self._init_args[param_names[i]] = arg
        
        # Add keyword args
        self._init_args.update(kwargs)
        
        # Convert string dtype to torch.dtype if needed
        if 'dtype' in self._init_args and isinstance(self._init_args['dtype'], str):
            dtype_str = self._init_args['dtype'].lower()
            dtype_mapping = {
                'float32': torch.float32,
                'float': torch.float32,
                'float64': torch.float64,
                'double': torch.float64,
                'float16': torch.float16,
                'half': torch.float16,
                'bfloat16': torch.bfloat16
            }
            
            if dtype_str in dtype_mapping:
                # Update both the stored init args and the kwargs that will be passed to the original init
                self._init_args['dtype'] = dtype_mapping[dtype_str]
                kwargs['dtype'] = dtype_mapping[dtype_str]
            else:
                raise ValueError(f"Unsupported dtype string: {dtype_str}")
        
        # Call the original __init__
        original_init(self, *args, **kwargs)
    
    cls.__init__ = new_init
    return cls


def create_classifier(model_arch: str, **kwargs) -> nn.Module:
    """
    Factory function to create a classifier based on model architecture.
    
    Args:
        model_arch: String indicating the architecture of the classifier to create.
                   Should be a class name registered in the MODEL_REGISTRY.
        **kwargs: Additional arguments to pass to the classifier constructor
        
    Returns:
        An instance of the appropriate classifier
    """

    # First, check if input_type is directly in the registry (exact match)
    if model_arch in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_arch](**kwargs)
    
    # Then check for case-insensitive match
    input_type_lower = model_arch.lower()
    if input_type_lower in MODEL_REGISTRY:
        return MODEL_REGISTRY[input_type_lower](**kwargs)
    
    # If not found in registry, raise an error
    valid_options = list(
        set([name for name, cls in MODEL_REGISTRY.items() if name == cls.__name__])
    )  # Only include actual class names
    raise ValueError(
        f"Unknown model type: {model_arch}. Valid options are: {valid_options}"
    )


# Function to create a classifier from a JSON configuration
def create_classifier_from_json(config_path: str) -> nn.Module:
    """
    Create a classifier from a JSON configuration file.

    Args:
        config_path: Path to the JSON configuration file

    Returns:
        An instance of the appropriate classifier
    """
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extract model_type and kwargs from config
    model_type = config.pop("model_type", None)
    if model_type is None:
        raise ValueError("model_type is required in the configuration file")

    # Create classifier
    return create_classifier(model_type, **config)


def save_model(
    model: nn.Module,
    output_file: str | None = None,
    threshold: float | None = None,
    input_type: List[str] | None = None,
    **kwargs,
):
    """
    Save the trained PyTorch model with its configuration.

    Args:
        model: The model to save
        output_file: Path to save the model to. If None, a default path will be created.
        threshold: Optional threshold value for classification
        **kwargs: Additional architecture-specific arguments
    """
    # If no output file specified, create default path with timestamp
    if output_file is None:
        from datetime import datetime
        import os

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("output/model", exist_ok=True)
        output_file = f"output/model/classifier_{timestamp}.pt"

    threshold = float(threshold) if threshold is not None else None

    # Create model config dictionary
    model_config = {}

    # Determine model type from registry
    model_type = None
    for name, cls in MODEL_REGISTRY.items():
        if isinstance(model, cls) and name == cls.__name__:
            model_type = name
            break
    if model_type is None:
        # Fallback to class name if not found in registry
        model_type = model.__class__.__name__

    # Add model_type to config
    model_config["model_type"] = model_type

    # Initialize the three categories of parameters
    init_args = getattr(model, "_init_args", {})
    common_args = {"threshold": threshold, "input_type": input_type}
    model_specific_args = kwargs

    # Add the three categories to the model config
    model_config["init_args"] = init_args
    model_config["common_args"] = common_args
    model_config["model_specific_args"] = model_specific_args
    model_config["state_dict"] = model.state_dict()

    # Create output directory if it doesn't exist
    import os

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    # Save the model config with all parameters
    output_file_all = output_file.replace(".pt", "_all.pt")
    
    # Create output_file_all directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file_all), exist_ok=True)

    torch.save(model_config, output_file_all)
    
    # Remove token_embeddings.weight from state_dict
    if 'token_embeddings.weight' in model_config["state_dict"]:
        del model_config["state_dict"]['token_embeddings.weight']
        print(f"Removed 'token_embeddings.weight' from state_dict")
    else:
        print(f"No 'token_embeddings.weight' to remove from state_dict")
    
    # Save the model config with only the parameters trained
    torch.save(model_config, output_file)
    
    # Print information about the saved model
    threshold_info = f" with threshold {threshold}" if threshold is not None else ""
    kwargs_info = f" and {len(kwargs)} additional parameters" if kwargs else ""

    print(
        f"\nModel saved to {output_file} with model_type '{model_type}'{threshold_info}{kwargs_info}"
    )


def load_model(model_path: str, device: str = "cuda", **kwargs) -> tuple[nn.Module, dict]:
    """
    Load a model from a saved file.

    Args:
        model_path: Path to the saved model
        device: Device to load the model to

    Returns:
        The loaded model and its configuration
    """
    # Load model configuration
    model_config = torch.load(model_path, map_location=device, weights_only=False)

    # Extract model_type and state_dict
    model_type = model_config.pop("model_type", None)
    state_dict = model_config.pop("state_dict", None)

    # Extract the three categories of parameters
    init_args = model_config.pop("init_args", {})
    common_args = model_config.pop("common_args", {})
    model_specific_args = model_config.pop("model_specific_args", {})

    # Create model with init_args
    dtype = model_config.get("training", {}).pop("dtype", None)
    model = create_classifier(model_type, **init_args)

    # Load state dict
    if state_dict is not None:
        # Check if model has token_embeddings and if the state_dict is missing token_embeddings.weight
        if hasattr(model, 'token_embeddings') and 'token_embeddings.weight' not in state_dict:
            # Add it to the state_dict
            state_dict['token_embeddings.weight'] = model.token_embeddings.weight
            print(f"Added 'token_embeddings.weight' to state_dict")
        else:
            print(f"No 'token_embeddings.weight' to add to state_dict")
        
        model.load_state_dict(state_dict)
    else:
        raise ValueError("State dict is not found in the model configuration")

    model = model.to(device=device, dtype=dtype)

    # Return model and all config parameters
    return model, {
        "model_type": model_type,
        "init_args": init_args,
        "common_args": common_args,
        "model_specific_args": model_specific_args,
    }


################ Model Architecture #################


class ClassifierBlock(nn.Module):
    """
    A single transformer-style block for the classifier backbone.
    Implements layer normalization, MLP with expansion, and residual connections.
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        expansion_factor=4,
        dropout_rate=0.3
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Pre-layer normalization
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # MLP with expansion
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * expansion_factor),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(input_dim * expansion_factor, output_dim),
            nn.Dropout(dropout_rate),
        )
        
        # Dimension change projection if needed
        self.dim_change = (
            nn.Linear(input_dim, output_dim)
            if input_dim != output_dim
            else nn.Identity()
        )
        
    def forward(self, x):
        # Apply layer norm
        normalized = self.layer_norm(x)
        
        # Handle dimension change for residual if needed
        residual = self.dim_change(x)
        
        # MLP block with residual connection
        return residual + self.mlp(normalized)


class ClassifierBackbone(nn.Module):
    """
    Backbone architecture for all classifiers.
    Implements transformer-style MLP blocks with residual connections.
    """

    def __init__(
        self,
        input_dim,
        output_dim=1,  # Default to 1 for binary classification
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
    ):
        super().__init__()
        self.use_position_embedding = use_position_embedding
        
        if self.use_position_embedding:
            # Generate sinusoidal position encoding and register as buffer
            position_embedding = get_sinusoidal_position_embedding(
                max_position_embeddings, hidden_dims[0]
            )
            self.register_buffer("position_embedding", position_embedding)
        
        # Build transformer-style blocks
        self.blocks = nn.ModuleList()
        
        # First projection to match the first hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        block_dims = hidden_dims + [hidden_dims[-1]] # for the last block, the output dimension is the same as the input dimension

        # Create classifier blocks with proper input and output dimensions
        for i in range(len(block_dims) - 1):
            # For the last block, output dimension is the same as input
            # For other blocks, output dimension is the next hidden dimension
            block_input_dim = block_dims[i]
            block_output_dim = block_dims[i+1]
            
            # Create a single block
            self.blocks.append(
                ClassifierBlock(
                    input_dim=block_input_dim,
                    output_dim=block_output_dim,
                    expansion_factor=expansion_factor,
                    dropout_rate=dropout_rate,
                )
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dims[-1]),
            nn.Linear(hidden_dims[-1], output_dim),
        )
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def apply_position_embedding(self, x, position_ids=None):
        """Apply position embeddings to the input if enabled."""
        if self.use_position_embedding:
            if position_ids is None:
                # Use default position ids (0, 1, 2, ...)
                batch_size = x.shape[0]
                seq_length = x.shape[1] if len(x.shape) > 2 else 1
                position_ids = torch.arange(seq_length, device=x.device).expand(
                    (batch_size, -1)
                )
            
            # Get position embeddings
            pos_emb = self.position_embedding[position_ids]
            
            # Add position embeddings to input
            x = x + pos_emb
        
        return x
    
    def forward(self, x, position_ids=None):
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        # Project input to first hidden dimension
        x = self.input_projection(x)
        
        # Apply position embeddings if enabled
        x = self.apply_position_embedding(x, position_ids)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output layer
        return self.output_layer(x)


@register_model
@capture_init_args
class LogitsClassifier(nn.Module):
    """
    A classifier that only takes logits as input.
    """

    def __init__(
        self,
        logits_size,
        output_dim=1,
        hidden_dims=[256, 512, 256],
        expansion_factor=4,
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        normalize_input=False,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        
        # Input projection for logits
        self.logits_projection = nn.Linear(logits_size, hidden_dims[0])
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )
    
    def forward(self, logits, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            logits: Tensor of logits
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        
        # Apply softmax normalization if enabled
        if self.normalize_input:
            logits = torch.nn.functional.softmax(logits, dim=-1)
            
        x = self.logits_projection(logits)
        
        # Process through backbone
        return self.backbone(x, position_ids)


@register_model
@capture_init_args
class HiddenStatesClassifier(nn.Module):
    """
    A classifier that only takes hidden states as input.
    """

    def __init__(
        self,
        hidden_states_size,
        output_dim=1,
        hidden_dims=[256, 512, 256],
        expansion_factor=4,
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        normalize_input=False,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        
        # Layer normalization for hidden states if normalization is enabled
        if self.normalize_input:
            self.layer_norm = nn.LayerNorm(hidden_states_size)
        
        # Input projection for hidden states
        self.hidden_states_projection = nn.Linear(
            hidden_states_size, hidden_dims[0]
        )
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )
    
    def forward(self, hidden_states, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Tensor of hidden states
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        
        # Apply layer normalization if enabled
        if self.normalize_input:
            hidden_states = self.layer_norm(hidden_states)
            
        x = self.hidden_states_projection(hidden_states)
        
        # Process through backbone
        return self.backbone(x, position_ids)


@register_model
@capture_init_args
class HiddenStatesClassifierWithLMHead(nn.Module):
    """
    A classifier that uses a pretrained LM head for processing hidden states.
    This model only accepts hidden states as input and does not use logits.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        hidden_states_size=1536,
        output_dim=1,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_position_embedding=False,
        max_position_embeddings=1024,
        topk=None,
        normalize_input=False,
    ):  # Add topk parameter
        super().__init__()
        self.topk = topk  # Store topk parameter
        self.pretrained_model_name = pretrained_model_name
        self.normalize_input = normalize_input
        
        # Layer normalization for hidden states if normalization is enabled
        if self.normalize_input:
            self.layer_norm = nn.LayerNorm(hidden_states_size)
        
        # Copy weights from a pretrained LM head
        try:
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            lm_head_weight = pretrained_model.get_output_embeddings().weight
            lm_head_bias = (
                pretrained_model.get_output_embeddings().bias
                if hasattr(pretrained_model.get_output_embeddings(), "bias")
                else None
            )
            
            # Create LM head with pretrained weights
            self.lm_head = nn.Linear(
                hidden_states_size, lm_head_weight.shape[0]
            )
            with torch.no_grad():
                self.lm_head.weight.copy_(lm_head_weight)
                if lm_head_bias is not None:
                    self.lm_head.bias.copy_(lm_head_bias)
            
            print(f"Successfully copied weights from {pretrained_model_name} LM head")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            # Fallback to random initialization
            self.lm_head = nn.Linear(hidden_states_size, hidden_dims[0])
            print(f"Using randomly initialized LM head")
        
        # Post-LM projection - adjust input size based on whether we're using topk
        post_lm_input_size = (
            self.topk if self.topk is not None else self.lm_head.out_features
        )
        self.post_lm_projection = nn.Linear(
            post_lm_input_size, hidden_dims[0]
        )
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )
        
    def _init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Linear) and module is not self.lm_head
            ):  # Skip LM head which is pretrained
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, hidden_states, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Tensor of hidden states
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        # Process hidden states through LM head
        
        # Apply layer normalization if enabled
        if self.normalize_input:
            hidden_states = self.layer_norm(hidden_states)
            
        lm_output = self.lm_head(hidden_states)
        
        # Apply topk if specified
        if self.topk is not None:
            # Get the top k values and sort them in descending order
            topk_values, _ = torch.topk(lm_output, k=self.topk, dim=-1)
            lm_output = torch.sort(topk_values, dim=-1, descending=True)[0]
        
        # Project LM output to the desired dimension
        x = self.post_lm_projection(lm_output)
        
        # Process through backbone
        return self.backbone(x, position_ids)



@register_model
@capture_init_args
class HiddenStatesTokenLMHeadClassifier(nn.Module):
    """
    A classifier that uses a pretrained LM head for processing token.
    This model accepts hidden states, sampled token as input.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        hidden_states_size=1536,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_position_embedding=False,
        max_position_embeddings=1024,
        topk=None,
        normalize_input=False,
        freeze_lm_head=False,
    ):  # Add topk parameter
        super().__init__()
        self.topk = topk  # Store topk parameter
        self.pretrained_model_name = pretrained_model_name
        self.normalize_input = normalize_input
        self.freeze_lm_head = freeze_lm_head
        
        # Layer normalization for hidden states if normalization is enabled
        if self.normalize_input:
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size)
            self.layer_norm_token = nn.LayerNorm(hidden_states_size)
        
        # Copy weights from a pretrained LM head
        try:
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            
            # Get the token embeddings from the pretrained model and create a copy
            embedding_layer = pretrained_model.get_input_embeddings()
            embed_dim = embedding_layer.embedding_dim

            self.token_embeddings = nn.Embedding(
                embedding_layer.num_embeddings,
                embed_dim,
            )
            with torch.no_grad():     
                self.token_embeddings.weight.copy_(embedding_layer.weight)
            
            print(f"Successfully copied weights from {pretrained_model_name} LM head and embeddings")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            # Create a simple embedding layer as fallback
            vocab_size = 50257  # Default GPT-2 vocab size
            embed_dim = hidden_dims[0]
            self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
            print(f"Using randomly initialized LM head and embeddings")

        if self.freeze_lm_head:
            self.token_embeddings.weight.requires_grad = False
                
        # Combined projection for hidden states + token embeddings
        combined_size = hidden_states_size + embed_dim
        self.combined_projection = nn.Linear(
            combined_size, hidden_dims[0]
        )
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=1,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )
        
    def _init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Linear) and module is not self.lm_head
            ):  # Skip LM head which is pretrained
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, hidden_states, token=None, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Tensor of hidden states [batch_size, hidden_dim]
            token: Tensor of token IDs [batch_size]
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        
        # Get token embeddings
        token_embeddings = self.token_embeddings(token)

        # Apply layer normalization if enabled
        if self.normalize_input:
            hidden_states = self.layer_norm_hidden_states(hidden_states)
            token_embeddings = self.layer_norm_token(token_embeddings)
        
        # Concatenate hidden states with token embeddings
        combined_features = torch.cat([hidden_states, token_embeddings], dim=-1)
        
        # Project the combined features
        x = self.combined_projection(combined_features)

        # Process through backbone
        return self.backbone(x, position_ids)


@register_model
@capture_init_args
class HiddenStatesTokenLMHeadLogitsClassifier(nn.Module):
    """
    A classifier that uses a pretrained LM head for processing token and also takes logits as input.
    This model accepts hidden states, sampled token, and logits as input.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        hidden_states_size=1536,
        logits_size=100,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_position_embedding=False,
        max_position_embeddings=1024,
        topk=None,
        normalize_input=False,
        freeze_lm_head=False,
    ):
        super().__init__()
        self.topk = topk  # Store topk parameter
        self.pretrained_model_name = pretrained_model_name
        self.normalize_input = normalize_input
        self.freeze_lm_head = freeze_lm_head
        
        # Layer normalization for inputs if normalization is enabled
        if self.normalize_input:
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size)
            self.layer_norm_token = nn.LayerNorm(hidden_states_size)
            # No layer norm for logits as we'll apply softmax instead if normalize_input is True
        
        # Copy weights from a pretrained LM head
        try:
            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            
            # Get the token embeddings from the pretrained model and create a copy
            embedding_layer = pretrained_model.get_input_embeddings()
            embed_dim = embedding_layer.embedding_dim

            self.token_embeddings = nn.Embedding(
                embedding_layer.num_embeddings,
                embed_dim
            )
            with torch.no_grad():     
                self.token_embeddings.weight.copy_(embedding_layer.weight)
            
            print(f"Successfully copied weights from {pretrained_model_name} embeddings")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            # Create a simple embedding layer as fallback
            vocab_size = 50257  # Default GPT-2 vocab size
            embed_dim = hidden_dims[0]
            self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
            print(f"Using randomly initialized embeddings")

        if self.freeze_lm_head:
            self.token_embeddings.weight.requires_grad = False
        
        # Projection for logits
        self.logits_projection = nn.Linear(logits_size, hidden_states_size)
        
        # Combined projection for hidden states + token embeddings + logits projection
        combined_size = hidden_states_size + embed_dim + hidden_states_size
        self.combined_projection = nn.Linear(
            combined_size, hidden_dims[0]
        )
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=1,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.token_embeddings:
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, hidden_states, token=None, logits=None, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Tensor of hidden states [batch_size, hidden_dim]
            token: Tensor of token IDs [batch_size]
            logits: Tensor of logits [batch_size, logits_size]
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        
        # Get token embeddings
        token_embeddings = self.token_embeddings(token)

        # Apply normalization if enabled
        if self.normalize_input:
            hidden_states = self.layer_norm_hidden_states(hidden_states)
            token_embeddings = self.layer_norm_token(token_embeddings)
            logits = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process logits
        logits_features = self.logits_projection(logits)
        
        # Concatenate all features
        combined_features = torch.cat([hidden_states, token_embeddings, logits_features], dim=-1)
        
        # Project the combined features
        x = self.combined_projection(combined_features)

        # Process through backbone
        return self.backbone(x, position_ids)


@register_model
@capture_init_args
class HiddenStatesLogitsClassifier(nn.Module):
    """
    A classifier that uses a pretrained LM head for processing token and also takes logits as input.
    This model accepts hidden states, sampled token, and logits as input.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        hidden_states_size=1536,
        logits_size=100,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        
        normalize_input=False,
        freeze_lm_head=False,
    ):
        super().__init__()
        
        self.normalize_input = normalize_input
        self.freeze_lm_head = freeze_lm_head
        
        # Layer normalization for inputs if normalization is enabled
        if self.normalize_input:
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size)
            # No layer norm for logits as we'll apply softmax instead if normalize_input is True
       
        
        # Projection for logits
        self.logits_projection = nn.Linear(logits_size, hidden_states_size)
        
        # Combined projection for hidden states +  logits projection
        combined_size = hidden_states_size + hidden_states_size
        self.combined_projection = nn.Linear(
            combined_size, hidden_dims[0]
        )
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=1,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(self, hidden_states, logits, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Tensor of hidden states [batch_size, hidden_dim]
            logits: Tensor of logits [batch_size, logits_size]
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        # Apply normalization if enabled
        if self.normalize_input:
            hidden_states = self.layer_norm_hidden_states(hidden_states)
            logits = torch.nn.functional.softmax(logits, dim=-1)
        
        # Process logits
        logits_features = self.logits_projection(logits)
        
        # Concatenate all features
        combined_features = torch.cat([hidden_states,  logits_features], dim=-1)
        
        # Project the combined features
        x = self.combined_projection(combined_features)

        # Process through backbone
        return self.backbone(x, position_ids)


@register_model
@capture_init_args
class HiddenStatesTokenClassifierWithLMHead(nn.Module):
    """
    A classifier that uses a pretrained LM head for processing hidden states and token.
    This model accepts hidden states, sampled token as input.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        hidden_states_size=1536,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_position_embedding=False,
        max_position_embeddings=1024,
        topk=None,
        normalize_input=False,
    ):  # Add topk parameter
        super().__init__()
        raise NotImplementedError("This model is not implemented yet")
        self.topk = topk  # Store topk parameter
        self.pretrained_model_name = pretrained_model_name
        self.normalize_input = normalize_input
        
        # Layer normalization for hidden states if normalization is enabled
        if self.normalize_input:
            self.layer_norm_hidden_states = nn.LayerNorm(hidden_states_size)
            self.layer_norm_token = nn.LayerNorm(hidden_states_size)
        
        # Copy weights from a pretrained LM head
        try:
            from transformers import AutoModelForCausalLM

            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            lm_head_weight = pretrained_model.get_output_embeddings().weight
            lm_head_bias = (
                pretrained_model.get_output_embeddings().bias
                if hasattr(pretrained_model.get_output_embeddings(), "bias")
                else None
            )
            
            # Get the token embeddings from the pretrained model and create a copy
            embedding_layer = pretrained_model.get_input_embeddings()
            embed_dim = embedding_layer.embedding_dim

            self.token_embeddings = nn.Embedding(
                embedding_layer.num_embeddings,
                embed_dim
            )
            self.lm_head = nn.Linear(
                hidden_states_size, lm_head_weight.shape[0]
            )
            with torch.no_grad():
                self.lm_head.weight.copy_(lm_head_weight)
                if lm_head_bias is not None:
                    self.lm_head.bias.copy_(lm_head_bias)            
                self.token_embeddings.weight.copy_(embedding_layer.weight)
            
            print(f"Successfully copied weights from {pretrained_model_name} LM head and embeddings")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            # Fallback to random initialization
            self.lm_head = nn.Linear(hidden_states_size, hidden_dims[0])
            # Create a simple embedding layer as fallback
            vocab_size = 50257  # Default GPT-2 vocab size
            embed_dim = hidden_dims[0]
            self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
            print(f"Using randomly initialized LM head and embeddings")
        
        # Post-LM projection - adjust input size based on whether we're using topk
        post_lm_input_size = (
            self.topk if self.topk is not None else self.lm_head.out_features
        )
        self.post_lm_projection = nn.Linear(
            post_lm_input_size, hidden_dims[0]
        )
        
        # Combined projection for hidden states + token embeddings
        combined_size = hidden_states_size + embed_dim
        self.combined_projection = nn.Linear(
            combined_size, hidden_dims[0]
        )
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )
        
    def _init_weights(self):
        for module in self.modules():
            if (
                isinstance(module, nn.Linear) and module is not self.lm_head
            ):  # Skip LM head which is pretrained
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, hidden_states, token=None, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            hidden_states: Tensor of hidden states [batch_size, hidden_dim]
            token: Tensor of token IDs [batch_size]
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        
        # Get token embeddings
        token_embeddings = self.token_embeddings(token)

        # Apply layer normalization if enabled
        if self.normalize_input:
            hidden_states = self.layer_norm_hidden_states(hidden_states)
            token_embeddings = self.layer_norm_token(token_embeddings)
        
        # Concatenate hidden states with token embeddings
        combined_features = torch.cat([hidden_states, token_embeddings], dim=-1)
        
        # Project the combined features
        x = self.combined_projection(combined_features)

        # Process through backbone
        return self.backbone(x, position_ids)




@register_model
@capture_init_args
class MultiLogitsClassifier(nn.Module):
    """
    Classifier that takes multiple logits inputs with a neural window.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        logits_size,  # logits_size = neural_window_size * single logit size
        output_dim=1,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        neural_window_size=3,  # Number of tokens to consider for classification
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        normalize_input=False,
    ):
        super().__init__()
        self.neural_window_size = neural_window_size
        self.normalize_input = normalize_input
        
        # Input projection for logits
        self.logits_projection = nn.Linear(logits_size, hidden_dims[0])
        
        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(self, logits, position_ids=None):
        """
        Forward pass through the model.
        
        Args:
            logits: Tensor of logits
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits)
        """
        
        # Apply softmax normalization if enabled
        if self.normalize_input:
            # For MultiLogitsClassifier, we need to reshape to apply softmax correctly
            # Assuming logits is [batch_size, neural_window_size * vocab_size]
            batch_size = logits.shape[0]
            single_logit_size = logits.shape[1] // self.neural_window_size
            
            # Reshape to [batch_size, neural_window_size, vocab_size]
            reshaped_logits = logits.view(batch_size, self.neural_window_size, single_logit_size)
            
            # Apply softmax on the vocab dimension
            normalized_logits = torch.nn.functional.softmax(reshaped_logits, dim=-1)
            
            # Reshape back to original shape
            logits = normalized_logits.reshape(batch_size, -1)
            
        x = self.logits_projection(logits)

        # Process through backbone
        return self.backbone(x, position_ids)


@register_model
@capture_init_args
class MultiHiddenStatesClassifier(nn.Module):
    """
    Classifier that takes multiple hidden states inputs with a neural window.
    Uses the ClassifierBackbone for processing.
    """

    def __init__(
        self,
        hidden_states_size=1024,
        output_dim=1,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        neural_window_size=3,  # Number of tokens to consider for classification
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        normalize_input=False,
    ):
        super().__init__()
        self.neural_window_size = neural_window_size
        self.normalize_input = normalize_input
        
        # Layer normalization for hidden states if normalization is enabled
        if self.normalize_input:
            self.layer_norm = nn.LayerNorm(hidden_states_size)

        # Input projection for hidden states
        self.hidden_states_projection = nn.Linear(
            hidden_states_size, hidden_dims[0]
        )

        # Create backbone
        self.backbone = ClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
        )

    def forward(self, hidden_states, position_ids=None):
        """
        Forward pass through the model.

        Args:
            hidden_states: Tensor of hidden states
            position_ids: Optional tensor of position ids for position embeddings

        Returns:
            Model output (logits)
        """
        
        # Apply layer normalization if enabled
        if self.normalize_input:
            # For MultiHiddenStatesClassifier, we need to handle the neural window dimension
            # Assuming hidden_states is [batch_size, neural_window_size * hidden_dim]
            batch_size = hidden_states.shape[0]
            single_hidden_size = hidden_states.shape[1] // self.neural_window_size
            
            # Reshape to [batch_size, neural_window_size, hidden_dim]
            reshaped_hidden = hidden_states.view(batch_size, self.neural_window_size, single_hidden_size)
            
            # Apply layer norm on each hidden state separately
            normalized_hidden = torch.stack([self.layer_norm(reshaped_hidden[:, i, :]) 
                                           for i in range(self.neural_window_size)], dim=1)
            
            # Reshape back to original shape
            hidden_states = normalized_hidden.reshape(batch_size, -1)
            
        x = self.hidden_states_projection(hidden_states)
        
        # Process through backbone
        return self.backbone(x, position_ids)

################ Threshold Optimization #################


class ThresholdOptimizer:
    """Handles threshold optimization and metric calculations."""

    def __init__(self, y_true, probs):
        self.y_true = y_true
        self.probs = probs
        self.total_samples = len(y_true)
        self.results = None
        
    def calculate_metrics(self, threshold):
        """Calculate metrics for a given threshold."""
        preds = (self.probs >= threshold).astype(
            float
        )  # pred = 1 is critical which means similarity <= similarity_threshold
        tn, fp, fn, tp = confusion_matrix(self.y_true, preds).ravel()
        
        return {
            "threshold": threshold,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "fp": fp,
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp_rate": fp / self.total_samples,
        }

    def calculate_metrics_hysteresis(self, threshold, window_size=3):
        """Calculate metrics for a given threshold."""
        critical_history = deque(maxlen=window_size)
        preds = []
        last_pred = "quick"
        for prob in self.probs:
            is_critical = (
                prob > threshold
            )  # prob > threshold: is_critical = 1, use reference model, pred = 1, similarity <= similarity_threshold
            is_simple = not is_critical
            if last_pred == "quick":
                pred = "quick" if is_simple else "reference"
                if pred == "reference":
                    critical_history.clear()
                    critical_history.append(prob)
            else:
                critical_history.append(prob)
                if len(critical_history) == 0:
                    pred = "reference"
                else:
                    avg_pred = sum(critical_history) / len(critical_history)
                    pred = "quick" if avg_pred <= threshold else "reference"
                if pred == "quick":
                    critical_history.clear()
            preds.append(pred)
            last_pred = pred
        predictions = [0 if pred == "quick" else 1 for pred in preds]
        predictions = torch.tensor(predictions)
        tn, fp, fn, tp = confusion_matrix(self.y_true, predictions).ravel()
        return {
            "threshold": threshold,
            "recall": tp / (tp + fn) if (tp + fn) > 0 else 0,
            "precision": tp / (tp + fp) if (tp + fp) > 0 else 0,
            "fp": fp,
            "tp": tp,
            "fn": fn,
            "tn": tn,
            "fp_rate": fp / self.total_samples,
        }

    def _find_best_compromise_threshold(self, min_recall, mode="default"):
        """Find best compromise threshold when no ideal solution exists."""
        thresholds = np.linspace(0.01, 0.99, 100)
        candidate_metrics = []
        
        for threshold in thresholds:
            metrics = self.calculate_metrics(threshold)
            if metrics["recall"] >= min_recall:
                candidate_metrics.append(metrics)

        if candidate_metrics:
            # save candidate_metrics
            results_df = pd.DataFrame(candidate_metrics)
            results_df.to_csv(f"candidate_metrics_{mode}.csv", index=False)
            print(f"Evaluation results saved to candidate_metrics_{mode}.csv")
            
            candidate_metrics.sort(key=lambda x: x["fp_rate"])
            self.results = candidate_metrics[0]
            # Convert to Python float directly
            threshold = float(self.results["threshold"])
            self.results["threshold"] = threshold
            return threshold
            
        print("Warning: No threshold meets minimum recall requirements")
        return 0.5
    
    def _find_best_compromise_threshold_hysteresis(self, min_recall, mode="default"):
        """Find best compromise threshold when no ideal solution exists."""
        thresholds = np.linspace(0.01, 0.99, 100)
        candidate_metrics = []
        
        for threshold in thresholds:
            metrics = self.calculate_metrics_hysteresis(threshold)
            if metrics["recall"] >= min_recall:
                candidate_metrics.append(metrics)

        if candidate_metrics:
            # save candidate_metrics
            results_df = pd.DataFrame(candidate_metrics)
            results_df.to_csv(f"candidate_metrics_{mode}.csv", index=False)
            print(f"Evaluation results saved to candidate_metrics_{mode}.csv")
            
            candidate_metrics.sort(key=lambda x: x["fp_rate"])
            self.results = candidate_metrics[0]
            # Convert to Python float directly
            threshold = float(self.results["threshold"])
            self.results["threshold"] = threshold
            return threshold
            
        print("Warning: No threshold meets minimum recall requirements")
        return 0.5
    
    def find_best_threshold(self, max_fp_rate=0.1, min_recall=0.9):
        """Find optimal threshold that meets constraints."""
        thresholds = np.linspace(0.01, 0.99, 100)
        valid_thresholds = []
        
        for threshold in thresholds:
            metrics = self.calculate_metrics(threshold)
            if metrics["recall"] >= min_recall and metrics["fp_rate"] <= max_fp_rate:
                valid_thresholds.append(metrics)

        if valid_thresholds:
            valid_thresholds.sort(key=lambda x: x["fp_rate"])
            self.results = valid_thresholds[0]
            # Convert to Python float directly
            threshold = float(self.results["threshold"])
            self.results["threshold"] = threshold
            return threshold
            
        threshold = self._find_best_compromise_threshold(min_recall, mode="default")
        # Convert to Python float directly
        threshold = float(threshold)
        return threshold

    def find_best_hysteresis_threshold(self, max_fp_rate=0.1, min_recall=0.9):
        """Find optimal threshold using hysteresis method."""
        thresholds = np.linspace(0.01, 0.99, 100)
        valid_thresholds = []
        
        for threshold in thresholds:
            metrics = self.calculate_metrics_hysteresis(threshold)
            if metrics["recall"] >= min_recall and metrics["fp_rate"] <= max_fp_rate:
                valid_thresholds.append(metrics)
                
        if valid_thresholds:
            valid_thresholds.sort(key=lambda x: x["fp_rate"])
            self.results = valid_thresholds[0]
            # Convert to Python float directly
            threshold = float(self.results["threshold"])
            self.results["threshold"] = threshold
            return threshold
            
        threshold = self._find_best_compromise_threshold_hysteresis(
            min_recall, mode="hysteresis"
        )
        # Convert to Python float directly
        threshold = float(threshold)
        return threshold

    def print_results(self):
        """Print optimization results with focus on recall and positive prediction rate."""
        if not self.results:
            raise ValueError("No results available. Run threshold optimization first.")
            
        # Calculate positive prediction rate
        total_positives = self.results["tp"] + self.results["fp"]
        positive_rate = (total_positives / self.total_samples) * 100
        
        print("\n" + "=" * 60)
        print("                 THRESHOLD OPTIMIZATION RESULTS                  ")
        print("=" * 60)
        print(f"OPTIMAL THRESHOLD:         {self.results['threshold']:.4f}")
        print(
            f"RECALL (True Positive Rate): {self.results['recall']:.4f} (Target: > 0.90)"
        )
        print(f"POSITIVE PREDICTION RATE:    {positive_rate:.2f}% (Target: < 10.00%)")
        print("=" * 60)
        
        print(f"\nAt threshold {self.results['threshold']:.4f}:")
        print(
            f"True Positives:  {self.results['tp']} ({self.results['recall']*100:.2f}% of critical cases)"
        )
        print(
            f"False Positives: {self.results['fp']} ({self.results['fp_rate']*100:.2f}% of total data)"
        )
        print(f"Precision:       {self.results['precision']:.4f}")
        
        # Provide guidance if targets aren't met
        if self.results["recall"] < 0.9:
            print("\nWARNING: Recall is below the target of 90%. Consider:")
            print("  - Decreasing the threshold value")
            print("  - Increasing class weights for the positive class")
            print("  - Using a higher alpha value in FocalLoss")
            
        if positive_rate > 10.0:
            print(
                "\nWARNING: Positive prediction rate exceeds the target of 10%. Consider:"
            )
            print("  - Increasing the threshold value")
            print("  - Adjusting the max_fp_rate parameter in find_best_threshold")


################ Utils #################

def get_sinusoidal_position_embedding(
    max_position_embeddings, embedding_dim, dtype=torch.float32
):
    """
    生成正弦位置编码
    
    Args:
        max_position_embeddings: 最大位置数
        embedding_dim: 嵌入维度
        dtype: 数据类型
        
    Returns:
        position_embeddings: 形状为 [max_position_embeddings, embedding_dim] 的位置编码
    """
    position = torch.arange(max_position_embeddings, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, embedding_dim, 2, dtype=dtype)
        * (-math.log(10000.0) / embedding_dim)
    )
    
    pe = torch.zeros(max_position_embeddings, embedding_dim, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe

class MultiClassClassifierBackbone(nn.Module):
    """
    Backbone architecture for multi-class classifiers.
    Uses the same ClassifierBlock as the regular backbone but with a multi-class output layer.
    """
    def __init__(
        self,
        input_dim,
        output_dim,  # Number of classes
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        apply_softmax=True,  # Whether to apply softmax at the output
    ):
        super().__init__()
        self.use_position_embedding = use_position_embedding
        self.apply_softmax = apply_softmax
        
        if self.use_position_embedding:
            # Generate sinusoidal position encoding and register as buffer
            position_embedding = get_sinusoidal_position_embedding(
                max_position_embeddings, hidden_dims[0]
            )
            self.register_buffer("position_embedding", position_embedding)
        
        # First projection to match the first hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dims[0])
        
        # Create transformer blocks with the new ClassifierBlock
        self.blocks = nn.ModuleList()
        
        # Create classifier blocks with proper input and output dimensions
        for i in range(len(hidden_dims)):
            # For the last block, output dimension is the same as input
            # For other blocks, output dimension is the next hidden dimension
            block_input_dim = hidden_dims[i]
            block_output_dim = hidden_dims[i] if i == len(hidden_dims) - 1 else hidden_dims[i + 1]
            
            self.blocks.append(
                ClassifierBlock(
                    input_dim=block_input_dim,
                    output_dim=block_output_dim,
                    expansion_factor=expansion_factor,
                    dropout_rate=dropout_rate,
                )
            )
        
        # Output layer for multi-class classification
        if self.apply_softmax:
            self.output_layer = nn.Sequential(
                nn.LayerNorm(hidden_dims[-1]),
                nn.Linear(hidden_dims[-1], output_dim),
                nn.Softmax(dim=-1)
            )
        else:
            self.output_layer = nn.Sequential(
                nn.LayerNorm(hidden_dims[-1]),
                nn.Linear(hidden_dims[-1], output_dim)
            )
            
    def apply_position_embedding(self, x, position_ids=None):
        """Apply position embeddings to the input if enabled."""
        if self.use_position_embedding:
            if position_ids is None:
                # Use default position ids (0, 1, 2, ...)
                batch_size = x.shape[0]
                seq_length = x.shape[1] if len(x.shape) > 2 else 1
                position_ids = torch.arange(seq_length, device=x.device).expand(
                    (batch_size, -1)
                )
            
            # Get position embeddings
            pos_emb = self.position_embedding[position_ids]
            
            # Add position embeddings to input
            x = x + pos_emb
        
        return x
    
    def forward(self, x, position_ids=None):
        """
        Forward pass through the backbone.
        
        Args:
            x: Input tensor
            position_ids: Optional tensor of position ids for position embeddings
            
        Returns:
            Model output (logits for each class)
        """
        # Project input to first hidden dimension
        x = self.input_projection(x)
        
        # Apply position embeddings if enabled
        x = self.apply_position_embedding(x, position_ids)
        
        # Process through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output layer
        return self.output_layer(x)


@register_model
@capture_init_args
class MultiClassLogitsClassifier(nn.Module):
    """
    A multi-class classifier that takes logits as input.
    """
    def __init__(
        self,
        logits_size,
        num_classes=3,
        hidden_dims=[256, 512, 256],
        expansion_factor=4,
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        normalize_input=False,
        apply_softmax=True,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.num_classes = num_classes
        
        # Input projection for logits
        self.logits_projection = nn.Linear(logits_size, hidden_dims[0])
        
        # Create multi-class backbone
        self.backbone = MultiClassClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
            apply_softmax=apply_softmax,
        )


@register_model
@capture_init_args
class MultiClassHiddenStatesClassifier(nn.Module):
    """
    A multi-class classifier that takes hidden states as input.
    """
    def __init__(
        self,
        hidden_states_size,
        num_classes=3,
        hidden_dims=[256, 512, 256],
        expansion_factor=4,
        dropout_rate=0.3,
        use_position_embedding=False,
        max_position_embeddings=1024,
        normalize_input=False,
        apply_softmax=True,
    ):
        super().__init__()
        self.normalize_input = normalize_input
        self.num_classes = num_classes
        
        # Layer normalization for hidden states if normalization is enabled
        if self.normalize_input:
            self.layer_norm = nn.LayerNorm(hidden_states_size)
        
        # Input projection for hidden states
        self.hidden_states_projection = nn.Linear(
            hidden_states_size, hidden_dims[0]
        )
        
        # Create multi-class backbone
        self.backbone = MultiClassClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
            apply_softmax=apply_softmax,
        )


@register_model
@capture_init_args
class MultiClassHiddenStatesClassifierWithLMHead(nn.Module):
    """
    A multi-class classifier that uses a pretrained LM head for processing hidden states.
    This model only accepts hidden states as input and does not use logits.
    Uses the MultiClassClassifierBackbone for processing.
    """
    def __init__(
        self,
        hidden_states_size=1536,
        num_classes=3,
        hidden_dims=[256, 512, 256],  # Bottleneck structure
        expansion_factor=4,  # Transformer-style expansion
        dropout_rate=0.3,
        pretrained_model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        use_position_embedding=False,
        max_position_embeddings=1024,
        topk=None,
        normalize_input=False,
        apply_softmax=True,
    ):
        super().__init__()
        self.topk = topk  # Store topk parameter
        self.pretrained_model_name = pretrained_model_name
        self.normalize_input = normalize_input
        self.num_classes = num_classes
        
        # Layer normalization for hidden states if normalization is enabled
        if self.normalize_input:
            self.layer_norm = nn.LayerNorm(hidden_states_size)
        
        # Copy weights from a pretrained LM head
        try:
            from transformers import AutoModelForCausalLM

            pretrained_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name
            )
            lm_head_weight = pretrained_model.get_output_embeddings().weight
            lm_head_bias = (
                pretrained_model.get_output_embeddings().bias
                if hasattr(pretrained_model.get_output_embeddings(), "bias")
                else None
            )
            
            # Create LM head with pretrained weights
            self.lm_head = nn.Linear(
                hidden_states_size, lm_head_weight.shape[0]
            )
            with torch.no_grad():
                self.lm_head.weight.copy_(lm_head_weight)
                if lm_head_bias is not None:
                    self.lm_head.bias.copy_(lm_head_bias)
            
            print(f"Successfully copied weights from {pretrained_model_name} LM head")
        except Exception as e:
            print(f"Failed to load pretrained model: {e}")
            # Fallback to random initialization
            self.lm_head = nn.Linear(hidden_states_size, hidden_dims[0])
            print(f"Using randomly initialized LM head")
        
        # Post-LM projection - adjust input size based on whether we're using topk
        post_lm_input_size = (
            self.topk if self.topk is not None else self.lm_head.out_features
        )
        self.post_lm_projection = nn.Linear(
            post_lm_input_size, hidden_dims[0]
        )
        
        # Create multi-class backbone
        self.backbone = MultiClassClassifierBackbone(
            input_dim=hidden_dims[0],
            output_dim=num_classes,
            hidden_dims=hidden_dims,
            expansion_factor=expansion_factor,
            dropout_rate=dropout_rate,
            use_position_embedding=use_position_embedding,
            max_position_embeddings=max_position_embeddings,
            apply_softmax=apply_softmax,
        )
