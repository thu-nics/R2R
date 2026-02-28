"""
Inference Backends Package

This package provides a pluggable backend system for different inference modes:
- R2R: Dynamic routing between Small Language Model (SLM) and Large Language Model (LLM)
- SingleLLM: Regular large language model inference (placeholder)

Usage:
    from backends import BackendRegistry, InferenceBackend
    
    # Register backends
    registry = BackendRegistry()
    registry.register("r2r", R2RBackend)
    registry.register("single_llm", SingleLLMBackend)
    
    # Create backend instance
    backend = registry.create("r2r", **config)
    
    # Generate response
    result = backend.generate("Your question here")
"""

from .base import InferenceBackend, GenerationResult, BackendConfig
from .registry import BackendRegistry
from .r2r_backend import R2RBackend
from .single_llm_backend import SingleLLMBackend

__all__ = [
    "InferenceBackend",
    "GenerationResult", 
    "BackendConfig",
    "BackendRegistry",
    "R2RBackend",
    "SingleLLMBackend",
]
