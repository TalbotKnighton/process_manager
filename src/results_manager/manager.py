from typing import Dict, Any, List, Type, Union, Optional, TypeVar, Generic
from pathlib import Path

from pydantic import BaseModel

from results_manager.backends.base import ResultsBackend, SetBehavior
from results_manager.backends.file_backend import FileBackend

T = TypeVar('T', bound=BaseModel)

class ResultsManager(Generic[T]):
    """
    Manages results from parallel processes, storing and retrieving pydantic models.
    
    This class provides a unified interface to different storage backends.
    """
    
    def __init__(self, 
                 base_dir: Union[str, Path] = None, 
                 create_if_missing: bool = True, 
                 backend: Optional[ResultsBackend] = None):
        """
        Initialize the ResultsManager.
        
        Args:
            base_dir: Base directory for file storage (used only if backend is None)
            create_if_missing: Whether to create the directory if it doesn't exist
            backend: Optional custom backend to use. If None, uses FileBackend.
        """
        if backend is None:
            if base_dir is None:
                raise ValueError("Must provide either base_dir or backend")
            self.backend = FileBackend(base_dir, create_if_missing)
        else:
            self.backend = backend
    
    def set(self, 
            result_id: Union[str, List[str]], 
            data: BaseModel, 
            behavior: SetBehavior = SetBehavior.RAISE_IF_EXISTS,
            namespace: Optional[str] = None,
            strict_namespace: bool = False) -> bool:
        """
        Store a result with the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            data: Pydantic model instance to store
            behavior: How to handle existing data with the same ID
            namespace: Optional namespace to store the model in. If None, will try to
                      determine the namespace from the model class automatically.
            strict_namespace: If True, raises an error if the model is registered 
                             in multiple non-default namespaces
            
        Returns:
            True if data was written, False if skipped (only for SKIP_IF_EXISTS)
            
        Raises:
            FileExistsError: If data already exists (for RAISE_IF_EXISTS) or
                             if different data exists (for RAISE_IF_DIFFERENT)
        """
        return self.backend.set(
            result_id=result_id, 
            data=data, 
            behavior=behavior, 
            namespace=namespace, 
            strict_namespace=strict_namespace
        )
    
    def get(self, 
            result_id: Union[str, List[str]], 
            model_class: Optional[Type[T]] = None,
            namespace: Optional[str] = None) -> T:
        """
        Retrieve a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            model_class: Optional model class to validate against. If not provided,
                         the stored model type will be used.
            namespace: Optional namespace override to look for the model in
                          
        Returns:
            Pydantic model instance
            
        Raises:
            FileNotFoundError: If the result doesn't exist
            ValueError: If the model type is not registered
            ValidationError: If the data doesn't match the model schema
        """
        return self.backend.get(
            result_id=result_id,
            model_class=model_class,
            namespace=namespace
        )
    
    def exists(self, result_id: Union[str, List[str]]) -> bool:
        """
        Check if a result exists for the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if the result exists, False otherwise
        """
        return self.backend.exists(result_id)
    
    def list_ids(self, prefix: Union[str, List[str]] = None) -> List[str]:
        """
        List all result IDs, optionally filtered by a prefix.
        
        Args:
            prefix: Optional prefix path to filter results
            
        Returns:
            List of result IDs
        """
        return self.backend.list_ids(prefix)
    
    def delete(self, result_id: Union[str, List[str]]) -> bool:
        """
        Delete a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if deleted, False if not found
        """
        return self.backend.delete(result_id)
    
    def clear(self) -> None:
        """
        Clear all stored results.
        """
        self.backend.clear()

     
    def _get_path_from_id(self, result_id: Union[str, List[str]]) -> Path:
        """
        Forward the path from ID call to the backend for testing purposes.
        
        Only works if the backend is a FileBackend.
        """
        if not hasattr(self.backend, '_get_path_from_id'):
            raise AttributeError(
                "_get_path_from_id is only available with FileBackend. "
                f"Current backend is {type(self.backend).__name__}"
            )
        return self.backend._get_path_from_id(result_id)
    
    def _get_backend_type(self) -> str:
        """Return the backend type name for testing."""
        return type(self.backend).__name__