"""
Registry for goodness functions.
Allows plug-and-play registration and retrieval of different goodness functions.
"""

from typing import Dict, Callable, Any

# Global registry to store goodness function builders
_GOODNESS_REGISTRY: Dict[str, Callable] = {}


def register(name: str):
    """
    Decorator to register a goodness function builder.
    
    Usage:
        @register("my_goodness")
        def build(cfg):
            return MyGoodnessClass(cfg)
    """
    def decorator(builder_fn: Callable) -> Callable:
        if name in _GOODNESS_REGISTRY:
            raise ValueError(f"Goodness function '{name}' is already registered!")
        _GOODNESS_REGISTRY[name] = builder_fn
        return builder_fn
    return decorator


def get_goodness_function(name: str, cfg: Any):
    """
    Retrieve and instantiate a goodness function by name.
    
    Args:
        name: Name of the registered goodness function
        cfg: Configuration object to pass to the builder
        
    Returns:
        Instantiated goodness function object
    """
    if name not in _GOODNESS_REGISTRY:
        available = list(_GOODNESS_REGISTRY.keys())
        raise ValueError(
            f"Goodness function '{name}' not found in registry. "
            f"Available functions: {available}"
        )
    return _GOODNESS_REGISTRY[name](cfg)


def list_available_goodness_functions():
    """Return list of all registered goodness function names."""
    return list(_GOODNESS_REGISTRY.keys())


def is_registered(name: str) -> bool:
    """Check if a goodness function is registered."""
    return name in _GOODNESS_REGISTRY
