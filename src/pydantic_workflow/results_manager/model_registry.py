from typing import Dict, Type, Optional, Any, List, Tuple
from pydantic import BaseModel

__all__ = [
    "register_model",
    "get_model_class",
    "find_model_namespace",
    "find_model_in_all_namespaces", 
    "clear_registry",
    "get_namespaces",
    "get_models_in_namespace",
    "DEFAULT_NAMESPACE",
]

# Registry structure: {namespace: {model_name: model_class}}
_MODEL_REGISTRY: Dict[str, Dict[str, Type[BaseModel]]] = {}
DEFAULT_NAMESPACE = "default"

def register_model(model_class_or_namespace: Any = None, *, namespace: str = DEFAULT_NAMESPACE):
    """
    Register a pydantic model class in the registry.
    
    Can be used as a decorator with or without arguments:
    
    @register_model
    class MyModel(BaseModel):
        ...
    
    @register_model(namespace="custom")
    class MyModel(BaseModel):
        ...
    
    Or programmatically:
    register_model(MyModel, namespace="custom")
    
    Args:
        model_class_or_namespace: The model class to register or a namespace string
        namespace: The namespace to register the model in (when used programmatically)
        
    Returns:
        The decorator function or the registered model class
    """
    # Handle case where register_model is called directly with a model class
    if isinstance(model_class_or_namespace, type) and issubclass(model_class_or_namespace, BaseModel):
        return _register_model(model_class_or_namespace, namespace)
    
    # Handle case where register_model is used as a decorator with or without arguments
    def decorator(model_class):
        if not isinstance(model_class, type) or not issubclass(model_class, BaseModel):
            raise TypeError("Registered model must be a subclass of BaseModel")
        
        # If model_class_or_namespace is a string, use it as namespace
        ns = model_class_or_namespace if isinstance(model_class_or_namespace, str) else namespace
        return _register_model(model_class, ns)
    
    return decorator

def _register_model(model_class: Type[BaseModel], namespace: str = DEFAULT_NAMESPACE) -> Type[BaseModel]:
    """
    Internal function to register a model class in a specific namespace.
    
    Args:
        model_class: The pydantic model class to register
        namespace: The namespace to register the model in
        
    Returns:
        The registered model class
    """
    model_name = model_class.__name__
    
    # Initialize namespace dictionary if it doesn't exist
    if namespace not in _MODEL_REGISTRY:
        _MODEL_REGISTRY[namespace] = {}
    
    _MODEL_REGISTRY[namespace][model_name] = model_class
    return model_class

def get_model_class(model_name: str, namespace: str = DEFAULT_NAMESPACE) -> Optional[Type[BaseModel]]:
    """
    Retrieve a model class from the registry by name and namespace.
    
    Args:
        model_name: The name of the model class
        namespace: The namespace to look in
        
    Returns:
        The model class if found, None otherwise
    """
    namespace_registry = _MODEL_REGISTRY.get(namespace, {})
    return namespace_registry.get(model_name)

# Update the find_model_namespace function

def find_model_namespace(model_class: Type[BaseModel], strict: bool = False) -> Optional[str]:
    """
    Find the namespace for a model class.
    
    If the model is registered in multiple namespaces, behavior depends on the 'strict' parameter:
    - If strict=False (default): Prioritizes non-default namespaces, returns the first one found
    - If strict=True: Raises ValueError if found in multiple non-default namespaces
    
    Args:
        model_class: The model class to find the namespace for
        strict: Whether to raise an error if the model is in multiple non-default namespaces
        
    Returns:
        The namespace name if found, None otherwise
        
    Raises:
        ValueError: If strict=True and the model is registered in multiple non-default namespaces
    """
    model_name = model_class.__name__
    found_namespaces = []
    
    # Find all namespaces containing this model class
    for namespace, models in _MODEL_REGISTRY.items():
        if model_name in models and models[model_name] is model_class:
            found_namespaces.append(namespace)
    
    if not found_namespaces:
        return None
    
    # Filter to just non-default namespaces
    non_default_namespaces = [ns for ns in found_namespaces if ns != DEFAULT_NAMESPACE]
    
    # If strict mode and multiple non-default namespaces, raise error
    if strict and len(non_default_namespaces) > 1:
        raise ValueError(
            f"Model '{model_name}' is registered in multiple non-default namespaces: "
            f"{', '.join(non_default_namespaces)}. Specify a namespace explicitly."
        )
    
    # Prioritize: first non-default namespace, or default namespace
    if non_default_namespaces:
        return non_default_namespaces[0]
    elif DEFAULT_NAMESPACE in found_namespaces:
        return DEFAULT_NAMESPACE
    else:
        return None  # Should not reach here, but just in case
def find_model_in_all_namespaces(model_name: str) -> List[Tuple[str, Type[BaseModel]]]:
    """
    Find a model by name in all namespaces.
    
    Args:
        model_name: The name of the model class
        
    Returns:
        List of (namespace, model_class) tuples for all matches
    """
    results = []
    for namespace, models in _MODEL_REGISTRY.items():
        if model_name in models:
            results.append((namespace, models[model_name]))
    return results

def clear_registry(namespace: Optional[str] = None):
    """
    Clear the model registry, optionally only for a specific namespace.
    
    Args:
        namespace: If provided, only clear this namespace. Otherwise, clear all.
    """
    if namespace is None:
        _MODEL_REGISTRY.clear()
    elif namespace in _MODEL_REGISTRY:
        _MODEL_REGISTRY[namespace].clear()

def get_namespaces() -> List[str]:
    """
    Get a list of all registered namespaces.
    
    Returns:
        List of namespace names
    """
    return list(_MODEL_REGISTRY.keys())

def get_models_in_namespace(namespace: str = DEFAULT_NAMESPACE) -> List[str]:
    """
    Get a list of all model names in a namespace.
    
    Args:
        namespace: The namespace to get models from
        
    Returns:
        List of model names
    """
    return list(_MODEL_REGISTRY.get(namespace, {}).keys())