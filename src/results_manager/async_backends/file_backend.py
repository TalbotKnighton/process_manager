import asyncio
from typing import Dict, Any, List, Type, Union, Optional, TypeVar, Generic
from pathlib import Path

from pydantic import BaseModel

from results_manager.async_backends.base import AsyncResultsBackend
from results_manager.backends.base import SetBehavior
from results_manager.backends.file_backend import FileBackend

T = TypeVar('T', bound=BaseModel)

class AsyncFileBackend(AsyncResultsBackend[T]):
    """
    Async wrapper for FileBackend.
    
    Runs the synchronous FileBackend methods in a threadpool to avoid blocking the event loop.
    """
    
    def __init__(self, base_dir: Union[str, Path], create_if_missing: bool = True, locks_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the AsyncFileBackend.
        
        Args:
            base_dir: Base directory to store results
            create_if_missing: Whether to create the directory if it doesn't exist
            locks_dir: Directory to store lock files. If None, uses a system temp directory.
        """
        # Create the synchronous backend
        self._backend = FileBackend(base_dir, create_if_missing, locks_dir)
    
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
        return await asyncio.to_thread(
            self._backend.set,
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
        return await asyncio.to_thread(
            self._backend.get,
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
        return await asyncio.to_thread(
            self._backend.exists,
            result_id=result_id
        )
    
    async def list_ids(self, prefix: Union[str, List[str]] = None) -> List[str]:
        """
        Asynchronously list all result IDs, optionally filtered by a prefix.
        
        Args:
            prefix: Optional prefix path to filter results
            
        Returns:
            List of result IDs
        """
        return await asyncio.to_thread(
            self._backend.list_ids,
            prefix=prefix
        )
    
    async def delete(self, result_id: Union[str, List[str]]) -> bool:
        """
        Asynchronously delete a result by ID.
        
        Args:
            result_id: Unique identifier or hierarchical path for the result
            
        Returns:
            True if deleted, False if not found
        """
        return await asyncio.to_thread(
            self._backend.delete,
            result_id=result_id
        )
    
    async def clear(self) -> None:
        """
        Asynchronously clear all stored results.
        """
        await asyncio.to_thread(
            self._backend.clear
        )