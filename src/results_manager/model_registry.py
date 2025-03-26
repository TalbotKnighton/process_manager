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
# src/results_manager/model_registry.py

def _register_model(model_class: Type[BaseModel], namespace: str = DEFAULT_NAMESPACE) -> Type[BaseModel]:
    """
    Internal function to register a model class in a specific namespace.
    
    Args:
        model_class: The pydantic model class to register
        namespace: The namespace to register the model in
        
    Returns:
        The registered model class
        
    Raises:
        ValueError: If a model with the same name but different structure is already registered
    """
    model_name = model_class.__name__
    
    # Initialize namespace dictionary if it doesn't exist
    if namespace not in _MODEL_REGISTRY:
        _MODEL_REGISTRY[namespace] = {}
    
    # Check if a model with this name already exists in this namespace
    if model_name in _MODEL_REGISTRY[namespace]:
        existing_model = _MODEL_REGISTRY[namespace][model_name]
        
        # Check if it's the exact same class (which is fine)
        if existing_model is model_class:
            return model_class
            
        # Check if field structure is the same (warning but allow)
        try:
            existing_fields = set(existing_model.model_fields.keys())
            new_fields = set(model_class.model_fields.keys())
            
            if existing_fields != new_fields:
                raise ValueError(
                    f"Model '{model_name}' is already registered in namespace '{namespace}' "
                    f"with different fields. Existing fields: {sorted(existing_fields)}, "
                    f"New fields: {sorted(new_fields)}"
                )
                
            # If field names match, check field types
            for field_name in existing_fields:
                existing_field = existing_model.model_fields[field_name]
                new_field = model_class.model_fields[field_name]
                
                # Check if field types are compatible
                # This is a simplistic check, might need to be enhanced
                if existing_field.annotation != new_field.annotation:
                    raise ValueError(
                        f"Model '{model_name}' is already registered in namespace '{namespace}' "
                        f"with different type for field '{field_name}'. "
                        f"Existing type: {existing_field.annotation}, "
                        f"New type: {new_field.annotation}"
                    )
            
            # If we get here, the models have the same structure but are different classes
            # We'll issue a warning but allow the replacement
            import warnings
            warnings.warn(
                f"Model '{model_name}' is already registered in namespace '{namespace}'. "
                f"Replacing with a new class with the same structure.",
                UserWarning
            )
            
        except Exception as e:
            if isinstance(e, ValueError) and str(e).startswith("Model '"):
                # Re-raise our custom error
                raise
            # If we can't compare the models properly, assume they're different
            raise ValueError(
                f"Model '{model_name}' is already registered in namespace '{namespace}' "
                f"and appears to have a different structure. Error: {str(e)}"
            )
    
    # Register the model
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