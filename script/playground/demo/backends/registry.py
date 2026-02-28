"""
Backend Registry - Factory for Managing Inference Backends

This module provides a centralized registry for creating and managing
different inference backends. It supports dynamic registration of new
backend types and provides factory methods for instantiation.
"""

from typing import Any, Callable, Dict, List, Optional, Type

from .base import InferenceBackend, BackendConfig, BackendType


# Type alias for backend factory functions
BackendFactory = Callable[[BackendConfig], InferenceBackend]


class BackendRegistry:
    """
    Registry for inference backends.
    
    The registry provides:
    1. Registration of backend types with their factory functions
    2. Creation of backend instances with configuration
    3. Discovery of available backends
    
    Usage:
        # Create registry and register backends
        registry = BackendRegistry()
        registry.register("r2r", R2RBackend)
        registry.register("single_llm", SingleLLMBackend)
        
        # Create backend instance
        config = BackendConfig(model_path="path/to/model")
        backend = registry.create("r2r", config)
        
        # Or use convenience method with keyword args
        backend = registry.create_with_kwargs("r2r", model_path="path/to/model")
    """
    
    def __init__(self):
        """Initialize an empty registry."""
        self._backends: Dict[str, Type[InferenceBackend]] = {}
        self._factories: Dict[str, BackendFactory] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        backend_class: Type[InferenceBackend],
        factory: Optional[BackendFactory] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a backend type.
        
        Args:
            name: Unique identifier for this backend type
            backend_class: The backend class (must inherit from InferenceBackend)
            factory: Optional custom factory function (defaults to calling class constructor)
            metadata: Optional metadata about this backend (description, version, etc.)
        
        Raises:
            ValueError: If name is already registered or backend_class is invalid
        """
        if name in self._backends:
            raise ValueError(f"Backend '{name}' is already registered")
        
        if not isinstance(backend_class, type) or not issubclass(backend_class, InferenceBackend):
            raise ValueError(
                f"backend_class must be a subclass of InferenceBackend, got {type(backend_class)}"
            )
        
        self._backends[name] = backend_class
        
        if factory is not None:
            self._factories[name] = factory
        
        self._metadata[name] = metadata or {}
        
        print(f"[BackendRegistry] Registered backend: {name}")
    
    def unregister(self, name: str) -> bool:
        """
        Unregister a backend type.
        
        Args:
            name: Name of the backend to unregister
            
        Returns:
            True if backend was unregistered, False if it wasn't registered
        """
        if name not in self._backends:
            return False
        
        del self._backends[name]
        self._factories.pop(name, None)
        self._metadata.pop(name, None)
        
        print(f"[BackendRegistry] Unregistered backend: {name}")
        return True
    
    def create(self, name: str, config: BackendConfig) -> InferenceBackend:
        """
        Create a backend instance.
        
        Args:
            name: Name of the backend type to create
            config: Configuration for the backend
            
        Returns:
            Initialized backend instance (not yet loaded)
            
        Raises:
            KeyError: If backend name is not registered
        """
        if name not in self._backends:
            available = ", ".join(self._backends.keys())
            raise KeyError(
                f"Backend '{name}' not registered. Available backends: {available}"
            )
        
        # Use custom factory if available, otherwise use class constructor
        if name in self._factories:
            return self._factories[name](config)
        else:
            return self._backends[name](config)
    
    def create_with_kwargs(self, name: str, **kwargs) -> InferenceBackend:
        """
        Create a backend instance with keyword arguments.
        
        Convenience method that creates a BackendConfig from kwargs.
        
        Args:
            name: Name of the backend type to create
            **kwargs: Configuration parameters (will be split into
                     BackendConfig fields and extra dict)
            
        Returns:
            Initialized backend instance
        """
        # Separate known BackendConfig fields from extra params
        config_fields = {'model_path', 'temperature', 'top_p', 'max_new_tokens', 'tp_size'}
        
        config_kwargs = {}
        extra_kwargs = {}
        
        for key, value in kwargs.items():
            if key in config_fields:
                config_kwargs[key] = value
            else:
                extra_kwargs[key] = value
        
        config_kwargs['extra'] = extra_kwargs
        config = BackendConfig(**config_kwargs)
        
        return self.create(name, config)
    
    def is_registered(self, name: str) -> bool:
        """Check if a backend type is registered."""
        return name in self._backends
    
    def get_backend_class(self, name: str) -> Type[InferenceBackend]:
        """
        Get the backend class for a registered type.
        
        Args:
            name: Name of the backend type
            
        Returns:
            The backend class
            
        Raises:
            KeyError: If backend name is not registered
        """
        if name not in self._backends:
            raise KeyError(f"Backend '{name}' not registered")
        return self._backends[name]
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get metadata for a registered backend."""
        return self._metadata.get(name, {})
    
    def list_backends(self) -> List[str]:
        """List all registered backend names."""
        return list(self._backends.keys())
    
    def get_backend_info(self) -> List[Dict[str, Any]]:
        """
        Get information about all registered backends.
        
        Returns:
            List of dicts with backend information
        """
        info = []
        for name, backend_class in self._backends.items():
            # Create a temporary instance to get display info
            # (without loading models)
            temp_config = BackendConfig()
            try:
                temp_instance = backend_class(temp_config)
                info.append({
                    "name": name,
                    "display_name": temp_instance.display_name,
                    "description": temp_instance.description,
                    "supports_colored_output": temp_instance.supports_colored_output(),
                    "metadata": self._metadata.get(name, {}),
                })
            except Exception:
                # Fallback if instance creation fails
                info.append({
                    "name": name,
                    "display_name": name,
                    "description": "No description available",
                    "metadata": self._metadata.get(name, {}),
                })
        return info
    
    def __contains__(self, name: str) -> bool:
        """Support 'in' operator for checking registration."""
        return name in self._backends
    
    def __len__(self) -> int:
        """Return number of registered backends."""
        return len(self._backends)


# Global default registry instance
_default_registry: Optional[BackendRegistry] = None


def get_default_registry() -> BackendRegistry:
    """
    Get the default global backend registry.
    
    Creates and populates the registry on first call.
    
    Returns:
        The default BackendRegistry instance
    """
    global _default_registry
    
    if _default_registry is None:
        _default_registry = BackendRegistry()
        _register_default_backends(_default_registry)
    
    return _default_registry


def _register_default_backends(registry: BackendRegistry) -> None:
    """Register all default backends with the registry."""
    from .r2r_backend import R2RBackend
    from .single_llm_backend import SingleLLMBackend, SingleLLMBackendWithR2REngine
    
    registry.register(
        "r2r",
        R2RBackend,
        metadata={
            "version": "1.0",
            "category": "routing",
            "gpu_required": True,
        }
    )
    
    registry.register(
        "single_llm",
        SingleLLMBackend,
        metadata={
            "version": "1.0",
            "category": "baseline",
            "gpu_required": True,
            "status": "placeholder",
        }
    )
    
    registry.register(
        "single_llm_shared",
        SingleLLMBackendWithR2REngine,
        metadata={
            "version": "1.0",
            "category": "baseline",
            "gpu_required": True,
            "requires": "r2r",
        }
    )


class BackendManager:
    """
    High-level manager for running multiple backends simultaneously.
    
    This class manages the lifecycle of multiple backends and provides
    convenient methods for parallel inference comparisons.
    
    Usage:
        manager = BackendManager()
        manager.add_backend("r2r", BackendConfig(extra={"router_path": "..."}))
        manager.add_backend("single_llm", BackendConfig(model_path="..."))
        
        manager.load_all()
        
        results = manager.generate_all("What is 2+2?")
        for name, result in results.items():
            print(f"{name}: {result.text}")
        
        manager.unload_all()
    """
    
    def __init__(self, registry: Optional[BackendRegistry] = None):
        """
        Initialize backend manager.
        
        Args:
            registry: Backend registry to use (defaults to global registry)
        """
        self.registry = registry or get_default_registry()
        self._backends: Dict[str, InferenceBackend] = {}
    
    def add_backend(self, name: str, config: BackendConfig) -> InferenceBackend:
        """
        Add a backend to the manager.
        
        Args:
            name: Name of the backend type to add
            config: Configuration for the backend
            
        Returns:
            The created backend instance
        """
        backend = self.registry.create(name, config)
        self._backends[name] = backend
        return backend
    
    def remove_backend(self, name: str) -> bool:
        """
        Remove a backend from the manager.
        
        Args:
            name: Name of the backend to remove
            
        Returns:
            True if removed, False if not found
        """
        if name not in self._backends:
            return False
        
        backend = self._backends.pop(name)
        backend.unload()
        return True
    
    def get_backend(self, name: str) -> Optional[InferenceBackend]:
        """Get a backend by name."""
        return self._backends.get(name)
    
    def load_all(self) -> Dict[str, bool]:
        """
        Load all backends.
        
        Returns:
            Dict mapping backend name to success status
        """
        results = {}
        for name, backend in self._backends.items():
            try:
                backend.load()
                results[name] = True
            except Exception as e:
                print(f"[BackendManager] Failed to load {name}: {e}")
                results[name] = False
        return results
    
    def unload_all(self) -> None:
        """Unload all backends."""
        for name, backend in self._backends.items():
            try:
                backend.unload()
            except Exception as e:
                print(f"[BackendManager] Error unloading {name}: {e}")
    
    def generate_all(self, prompt: str) -> Dict[str, Any]:
        """
        Generate responses from all loaded backends.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Dict mapping backend name to GenerationResult
        """
        from .base import GenerationResult
        
        results = {}
        for name, backend in self._backends.items():
            if backend.is_loaded:
                results[name] = backend.generate(prompt)
            else:
                results[name] = GenerationResult(
                    error=f"Backend {name} not loaded"
                )
        return results
    
    def list_backends(self) -> List[str]:
        """List names of all managed backends."""
        return list(self._backends.keys())
    
    def __enter__(self):
        """Context manager entry - load all backends."""
        self.load_all()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - unload all backends."""
        self.unload_all()
        return False
