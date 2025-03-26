from results_manager.manager import ResultsManager
from results_manager.backends.base import SetBehavior
from results_manager.model_registry import (
    register_model, get_model_class, clear_registry,
    DEFAULT_NAMESPACE, get_namespaces, get_models_in_namespace,
    find_model_namespace, find_model_in_all_namespaces
)
from .backends import ResultsBackend, FileBackend

# Import async components with proper error handling for optional dependencies
try:
    from results_manager.async_manager import AsyncResultsManager
    from results_manager.async_backends import AsyncResultsBackend, AsyncFileBackend
    _has_async = True
except ImportError:
    # Async support requires asyncio, which should be in standard library
    # But importing aiosqlite might fail
    _has_async = False

if _has_async:
    __all__ = [
        "ResultsManager", "AsyncResultsManager", 
        "SetBehavior", "ResultsBackend", "FileBackend", 
        "AsyncResultsBackend", "AsyncFileBackend",
        "register_model", "get_model_class", "clear_registry",
        "DEFAULT_NAMESPACE", "get_namespaces", "get_models_in_namespace",
        "find_model_namespace", "find_model_in_all_namespaces"
    ]
else:
    __all__ = [
        "ResultsManager", "SetBehavior", "ResultsBackend", "FileBackend",
        "register_model", "get_model_class", "clear_registry",
        "DEFAULT_NAMESPACE", "get_namespaces", "get_models_in_namespace",
        "find_model_namespace", "find_model_in_all_namespaces"
    ]