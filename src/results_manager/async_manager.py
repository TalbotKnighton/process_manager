from typing import Dict, Any, List, Type, Union, Optional, TypeVar, Generic
from pathlib import Path

from pydantic import BaseModel

from results_manager.model_registry import DEFAULT_NAMESPACE
from results_manager.backends.base import SetBehavior
from results_manager.async_backends.base import AsyncResultsBackend
from results_manager.async_backends.file_backend import AsyncFileBackend

T = TypeVar('T', bound=BaseModel)

class AsyncResultsManager(Generic[T]):
    """
    Async version of ResultsManager for managing results from parallel processes.
    
    Provides an asynchronous interface for storing and retrieving pydantic models.
    """
    
    def __init__(self, 
                 base_dir: Union[str, Path] = None, 
                 create_if_missing: bool = True, 
                 backend: Optional[AsyncResultsBackend] = None):
        """
        Initialize the AsyncResultsManager.
        
        Args:
            base_dir: Base directory for file storage (used only if backend is None)
            create_if_missing: Whether to create the directory if it doesn't exist
            backend: Optional custom async backend to use. If None, uses AsyncFileBackend.
        """
        if backend is None:
            if base_dir is None:
                raise ValueError("Must provide either base_dir or backend")
            self.backend = AsyncFileBackend(base_dir, create_if_missing)
        else:
            self.backend = backend
    
    async def set(self, 
                 result_id: Union[str, List[str]], 
                 data: BaseModel, 
                 behavior: SetBehavior = SetBehavior.RAISE_IF_EXISTS,
                 namespace: Optional[str] = None,
                 strict_namespace: bool = False) -> bool:
        """
        Asynchronously store a result with the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            data: Pydantic model instance to store
            behavior: How to handle existing data with the same ID
            namespace: Optional namespace to store the model in
            strict_namespace: If True, raises an error if model is in multiple namespaces
            
        Returns:
            True if data was written, False if skipped
        """
        return await self.backend.set(
            result_id=result_id, 
            data=data, 
            behavior=behavior, 
            namespace=namespace, 
            strict_namespace=strict_namespace
        )
    
    async def get(self, 
                 result_id: Union[str, List[str]], 
                 model_class: Optional[Type[T]] = None,
                 namespace: Optional[str] = None) -> T:
        """
        Asynchronously retrieve a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            model_class: Optional model class to validate against
            namespace: Optional namespace to look in
            
        Returns:
            Pydantic model instance
        """
        return await self.backend.get(
            result_id=result_id,
            model_class=model_class,
            namespace=namespace
        )
    
    async def exists(self, result_id: Union[str, List[str]]) -> bool:
        """
        Asynchronously check if a result exists for the given ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if the result exists, False otherwise
        """
        return await self.backend.exists(result_id)
    
    async def list_ids(self, prefix: Union[str, List[str]] = None) -> List[str]:
        """
        Asynchronously list all result IDs, optionally filtered by a prefix.
        
        Args:
            prefix: Optional prefix path to filter results
            
        Returns:
            List of result IDs
        """
        return await self.backend.list_ids(prefix)
    
    async def delete(self, result_id: Union[str, List[str]]) -> bool:
        """
        Asynchronously delete a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if deleted, False if not found
        """
        return await self.backend.delete(result_id)
    
    async def clear(self) -> None:
        """
        Asynchronously clear all stored results.
        """
        await self.backend.clear()