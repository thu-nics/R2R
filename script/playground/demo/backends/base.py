"""
Abstract Base Classes for Inference Backends

This module defines the interface that all inference backends must implement,
enabling seamless switching between different inference modes in the Gradio UI.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from enum import Enum


class BackendType(Enum):
    """Enumeration of supported backend types."""
    R2R = "r2r"
    SINGLE_LLM = "single_llm"
    # Future backends can be added here
    # SPECULATIVE = "speculative"
    # ENSEMBLE = "ensemble"


@dataclass
class BackendConfig:
    """
    Configuration for inference backends.
    
    This dataclass holds common configuration parameters shared across backends.
    Backend-specific parameters can be added via the `extra` field.
    
    Attributes:
        model_path: Path to the main model
        temperature: Sampling temperature (0.0 = deterministic)
        top_p: Top-p (nucleus) sampling parameter
        max_new_tokens: Maximum number of tokens to generate
        tp_size: Tensor parallelism size
        extra: Backend-specific additional parameters
    """
    model_path: Optional[str] = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_new_tokens: int = 4096
    tp_size: int = 2
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a parameter from extra config."""
        return self.extra.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set a parameter in extra config."""
        self.extra[key] = value


@dataclass
class TokenRecord:
    """
    Record of a single generated token.
    
    Attributes:
        token_id: The token's integer ID
        token_str: The decoded token string
        source_model: Name of the model that generated this token (e.g., "quick", "reference")
        probability: Generation probability (if available)
        metadata: Additional token-level metadata
    """
    token_id: int
    token_str: str
    source_model: str
    probability: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationStatistics:
    """
    Statistics about a generation run.
    
    Attributes:
        total_tokens: Total number of tokens generated
        elapsed_time_s: Total generation time in seconds
        tokens_per_second: Generation speed
        model_usage: Dict mapping model name to usage percentage
        avg_params_billions: Average model parameters used (for routing systems)
        extra: Additional backend-specific statistics
    """
    total_tokens: int = 0
    elapsed_time_s: float = 0.0
    tokens_per_second: float = 0.0
    model_usage: Dict[str, float] = field(default_factory=dict)
    avg_params_billions: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    
    def format_summary(self) -> str:
        """Format a human-readable summary of the statistics."""
        lines = [
            f"Tokens: {self.total_tokens}",
            f"Time: {self.elapsed_time_s:.2f}s",
            f"Speed: {self.tokens_per_second:.1f} tok/s",
        ]
        
        if self.model_usage:
            usage_str = " | ".join(
                f"{name}: {pct:.1f}%" 
                for name, pct in self.model_usage.items()
            )
            lines.append(usage_str)
        
        if self.avg_params_billions is not None:
            lines.append(f"Avg Params: {self.avg_params_billions:.2f}B")
        
        return " | ".join(lines)


@dataclass
class GenerationResult:
    """
    Result of a generation request.
    
    Attributes:
        text: The generated text (plain)
        html: HTML-formatted output (e.g., with colored tokens)
        statistics: Generation statistics
        token_records: List of individual token records (for detailed analysis)
        is_complete: Whether generation completed normally
        error: Error message if generation failed
    """
    text: str = ""
    html: str = ""
    statistics: GenerationStatistics = field(default_factory=GenerationStatistics)
    token_records: List[TokenRecord] = field(default_factory=list)
    is_complete: bool = True
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        """Check if generation was successful."""
        return self.error is None


class InferenceBackend(ABC):
    """
    Abstract base class for inference backends.
    
    All inference backends (R2R, Single LLM, etc.) must implement this interface
    to be compatible with the Gradio UI.
    
    Example implementation:
        class MyBackend(InferenceBackend):
            def __init__(self, config: BackendConfig):
                super().__init__(config)
                self._load_models()
            
            @property
            def name(self) -> str:
                return "my_backend"
            
            def generate(self, prompt: str) -> GenerationResult:
                # Implementation here
                ...
    """
    
    def __init__(self, config: BackendConfig):
        """
        Initialize the backend with configuration.
        
        Args:
            config: Backend configuration parameters
        """
        self.config = config
        self._is_loaded = False
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Return the unique name of this backend.
        
        Returns:
            String identifier for this backend type
        """
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """
        Return a human-readable display name for the UI.
        
        Returns:
            Human-readable name for display in UI
        """
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Return a brief description of what this backend does.
        
        Returns:
            Description string for UI help text
        """
        pass
    
    @property
    def is_loaded(self) -> bool:
        """Check if the backend models are loaded."""
        return self._is_loaded
    
    @abstractmethod
    def load(self) -> None:
        """
        Load the model(s) required for this backend.
        
        This method should be idempotent - calling it multiple times
        should not cause issues.
        
        Raises:
            RuntimeError: If model loading fails
        """
        pass
    
    @abstractmethod
    def unload(self) -> None:
        """
        Unload and cleanup model resources.
        
        This method should be safe to call even if load() was never called.
        """
        pass
    
    @abstractmethod
    def generate(self, prompt: str) -> GenerationResult:
        """
        Generate a response for the given prompt.
        
        This is the main generation method that returns a complete result.
        
        Args:
            prompt: The user's input prompt/question
            
        Returns:
            GenerationResult containing the generated text and metadata
        """
        pass
    
    def generate_stream(
        self, prompt: str
    ) -> Generator[GenerationResult, None, None]:
        """
        Generate a response with streaming output.
        
        Override this method to support streaming generation.
        Default implementation calls generate() once and yields the result.
        
        Args:
            prompt: The user's input prompt/question
            
        Yields:
            GenerationResult with incremental text updates
        """
        # Default: non-streaming fallback
        yield self.generate(prompt)
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded models.
        
        Returns:
            Dict with model information (names, sizes, paths, etc.)
        """
        pass
    
    def supports_colored_output(self) -> bool:
        """
        Check if this backend supports colored token output.
        
        Returns:
            True if generate() returns meaningful HTML with colored tokens
        """
        return False
    
    def get_color_legend(self) -> List[Tuple[str, str, str]]:
        """
        Get the color legend for token visualization.
        
        Returns:
            List of tuples: (color_hex, label, description)
        """
        return []
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters at runtime.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                self.config.extra[key] = value
    
    def __enter__(self):
        """Context manager entry - load models."""
        self.load()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload models."""
        self.unload()
        return False
